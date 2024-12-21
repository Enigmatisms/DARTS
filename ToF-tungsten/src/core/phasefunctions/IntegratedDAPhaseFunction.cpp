#include <mutex>
#include <fstream>
#include "IntegratedDAPhaseFunction.hpp"

#include "sampling/PathSampleGenerator.hpp"
#include "sampling/SampleWarp.hpp"

#include "math/TangentFrame.hpp"
#include "math/MathUtil.hpp"
#include "math/Angle.hpp"

#include "io/JsonObject.hpp"

// make sure tabulations will have correct behavior when fromJson is called
// by multiple different threads
static std::mutex table_mtx;
static std::unordered_map<std::string, float*> tabulations;
static constexpr int TABLE_SIZE = 128 * 128 * ANGULAR_RES;

static int binary_search_branchless_UR (const float *arr, float key) {
    // best latency performance, twice faster than std::lower_bound (yet when O3, the speed different is smaller)
    // this function returns the index of the first element that is greater or equal than the key
    // so this is equal to std::lower_bound
    intptr_t pos = -1;
    #define STEP(logstep) \
        pos += (arr[pos + (1<<logstep)] < key ? (1<<logstep) : 0);
    if constexpr (ANGULAR_RES > 128) {
        STEP(7)
    }
    STEP(6)
    STEP(5)
    STEP(4)
    STEP(3)
    STEP(2)
    STEP(1)
    STEP(0)
    #undef STEP
    return pos + 1;
}

namespace Tungsten {

IntegratedDAPhaseFunction::IntegratedDAPhaseFunction(): HenyeyGreensteinPhaseFunction(), 
    alpha(0.4f), dT_s(0.2), inv_dT_div(160.2), inv_max_T(0) {}

IntegratedDAPhaseFunction::~IntegratedDAPhaseFunction()
{
    // delete corresponding memory section (once)
    // be aware about multi-threading double-free problem
    {
        std::lock_guard lock(table_mtx);
        if (tabulations[table_path])
            delete [] tabulations[table_path];
    }
}

void IntegratedDAPhaseFunction::fromJson(JsonPtr value, const Scene &scene)
{
    HenyeyGreensteinPhaseFunction::fromJson(value, scene);

    // tabulation loading (multiple tables are allowed)
    std::string table_path;
    value.getField("table_path", table_path);
    {
        std::lock_guard lock(table_mtx);
        if (!tabulations.count(table_path))
            tabulations[table_path] = nullptr;
    }
    // only when the table is not loaded, will the table be loaded
    if (tabulations[table_path] == nullptr) {
        tabulations[table_path] = new (std::align_val_t(32)) float [TABLE_SIZE];        // align by a cache line
        std::ifstream file(table_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Unable to open binary table at " + table_path);
        }
        file.read(reinterpret_cast<char*>(tabulations[table_path]), sizeof(float) * TABLE_SIZE);
        printf("Tabulation data loaded from '%s'\n", table_path.c_str());
        file.close();
    }
    table = reinterpret_cast<float (*)[128][ANGULAR_RES]>(tabulations[table_path]);

    float r_s = 0.2, r_e = 0.999, max_T = 0;
    value.getField("alpha", alpha);
    value.getField("dT_start", r_s);
    value.getField("dT_end", r_e);
    value.getField("max_T", max_T);
    if (max_T == 0) {
        throw std::runtime_error("max_T for IntegratedDAPhase is not set, 0 is not acceptable.");
    }

    dT_s = r_s;
    inv_dT_div = 128.f / (r_e - r_s);
    inv_max_T = 1.f / max_T * 128.f;

    printf("Integrated DA. Alpha = %.2f, dT_s = %.3f, inv_dT_div = %.4f, inv_max_T = %.4f\n", alpha, dT_s, inv_dT_div, inv_max_T);
}

rapidjson::Value IntegratedDAPhaseFunction::toJson(Allocator &allocator) const
{
    return JsonObject{HenyeyGreensteinPhaseFunction::toJson(allocator), allocator,
        "type", "integrated_da",
        "table_path", table_path,
        "alpha", alpha,
        "dT_start", dT_s,
        "dT_end", 128.f / inv_dT_div + dT_s,
        "max_T", 128.f / inv_max_T
    };
}

static constexpr float WITHIN_BIN_PDF = static_cast<float>(ANGULAR_RES) / 2.f;        // 2 / 256 is interval length, pdf = 1 / interval length = 128

bool IntegratedDAPhaseFunction::sample(PathSampleGenerator &sampler, const Vec3f &wi, PhaseSample &phase_smp, EllipseConstPtr ell_info) const
{
    // MIS one sample model: choose between local phase function and ida
    // d/T should be one of the controlling factor
    // d_T / (d_T + alpha) = 1 / (T_d * alpha + 1) --- d/T adaptive strategy selection
    // this function is definitely much slower than HG::Sample_p
    // this function will sample cosTheta and phi, which will be of differential solid angle measure
    // no measure conversion needed here
    if (!ell_info || ell_info->T_div_D > (1.f / dT_s) || ell_info->T_div_D <= 1.050001f) 
        return HenyeyGreensteinPhaseFunction::sample(sampler, wi, phase_smp, nullptr);

    float ida_smp_proba = 1.f / (ell_info->T_div_D * alpha + 1);
    float cosTheta = 0, osm_pdf = 0, eval = 0;

    // tabulation look up
    int dT_index = (1.f / ell_info->T_div_D - dT_s) * inv_dT_div,     // convert to index
        T_index  = ell_info->target_t * inv_max_T, index = 0;
    const float* const ptr = table[dT_index][T_index];
    
    const bool osm_ida = sampler.next1D() < ida_smp_proba;
    if (osm_ida) {         // OSM IDA
        index = binary_search_branchless_UR(ptr, sampler.next1D());
        cosTheta  = (1.f / WITHIN_BIN_PDF) * (float(index) + sampler.next1D()) - 1; 
        float sinTheta = std::sqrt(std::max((float)0, 1.f - cosTheta * cosTheta));
        // HG sampling (cos theta and phi) is equivalent to sampling via differential solid angle measure
        float phi = TWO_PI * sampler.next1D();
        phase_smp.w = TangentFrame(ell_info->to_emitter).toGlobal(Vec3f(
            std::cos(phi)*sinTheta,
            std::sin(phi)*sinTheta,
            cosTheta
        ));
        eval = this->henyeyGreenstein(phase_smp.w.dot(wi));
    } else {  
        HenyeyGreensteinPhaseFunction::sample(sampler, wi, phase_smp);
        eval = phase_smp.pdf;
        float dot_value = std::clamp(phase_smp.w.dot(ell_info->to_emitter), -1.f, 0.99999f);
        index = int((dot_value + 1) * WITHIN_BIN_PDF);
    }
    // correct pdf p(within_bin | bin) * p(bin) * p(phi)
    osm_pdf   = (ptr[index] - (index ? ptr[index - 1] : 0)) * (WITHIN_BIN_PDF * INV_TWO_PI);
    // this is already MIS, returns weight * f / (c * pdf)
    // and since weight = pdf_i / (pdf_i + pdf_j), pdf used is pdf_i
    // therefore: f / (c * (pdf_i + pdf_j)) 
    float mis_pdf = (osm_ida ? ida_smp_proba : 1 - ida_smp_proba) * (osm_pdf + eval);
    phase_smp.weight = Vec3f(eval / mis_pdf);

    // for non-BDPT usage, PDF can left unfilled
    // phase_smp.pdf    = 1;
    phase_smp.pdf    = osm_ida ? ida_smp_proba * osm_pdf : (1 - ida_smp_proba) * eval;
    return true;
}

float IntegratedDAPhaseFunction::pdf(const Vec3f &wi, const Vec3f &wo, EllipseConstPtr ell_info) const {
    float hg_pdf = this->henyeyGreenstein(wi.dot(wo));
    if (!ell_info || ell_info->T_div_D > (1.f / dT_s) || ell_info->T_div_D < 1.05f) 
        return hg_pdf;

    int dT_index = (1 / ell_info->T_div_D - dT_s) * inv_dT_div,     // convert to index
        T_index  = ell_info->target_t * inv_max_T;
    const float* const ptr = table[dT_index][T_index];

    float ida_smp_proba = 1.f / (ell_info->T_div_D * alpha + 1),
          dot_value = std::clamp(wo.dot(ell_info->to_emitter), -1.f, 0.99999f);
    int index = int((dot_value + 1) * WITHIN_BIN_PDF);
    float osm_pdf   = ptr[index] - (index ? ptr[index - 1] : 0);
    osm_pdf  *= WITHIN_BIN_PDF * INV_TWO_PI;                
    return ida_smp_proba * osm_pdf + (1 - ida_smp_proba) * hg_pdf;
}

}
