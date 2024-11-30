import re
import sys

def replace_scattering_in_string(input_string, replacement_values):
    # Define the pattern with a regular expression
    pattern = re.compile(r'("rgb sigma_s" )\[([\d\.\s]+)\] ("rgb sigma_a")')
    # Find the match in the input string
    match = pattern.search(input_string)
    if match:
        # Retrieve the content inside the square brackets
        values_to_replace = match.group(2)
        # Replace the values with the new specified values
        updated_string = input_string.replace(values_to_replace, replacement_values)
        return updated_string
    else:
        return input_string  # Return the original string if no match is found
    
def scattering_experiments():
    float_num = float(sys.argv[2])
    with open(sys.argv[1], 'r') as file:
        all_info = file.read()
        output = replace_scattering_in_string(all_info, f"{float_num} {float_num} {float_num}")
    with open(sys.argv[1], 'w') as file:
        file.write(output)

def convergence_experiments():
    int_num = int(sys.argv[2])
    with open(sys.argv[1], 'r') as file:
        file_content = file.read()
    pattern = r'"integer pixelsamples" \[ (\d+) \]'
    replacement = fr'"integer pixelsamples" [ {int_num} ]'
    updated_content = re.sub(pattern, replacement, file_content)
    with open(sys.argv[1], 'w') as file:
        file.write(updated_content)
        
        
def gatewidth_experiments():
    float_val = float(sys.argv[2])
    with open(sys.argv[1], 'r') as file:
        file_content = file.read()
    pattern = r'"float t_interval" \[(.*?)\]'
    replacement = fr'"float t_interval" [{float_val}]'
    updated_content = re.sub(pattern, replacement, file_content)
    with open(sys.argv[1], 'w') as file:
        file.write(updated_content)

        
def double_ablation_experiments():  
    float_num = float(sys.argv[2])              # sigma_s
    float_val = float(sys.argv[3])
    with open(sys.argv[1], 'r') as file:
        all_info = file.read()
        output = replace_scattering_in_string(all_info, f"{float_num} {float_num} {float_num}")
    pattern = r'"float t_interval" \[(.*?)\]'
    replacement = fr'"float t_interval" [{float_val}]'
    updated_content = re.sub(pattern, replacement, output)
    with open(sys.argv[1], 'w') as file:
        file.write(updated_content)

def gatewidth_experiments():
    float_val = float(sys.argv[2])
    with open(sys.argv[1], 'r') as file:
        file_content = file.read()
    pattern = r'"float t_interval" \[(.*?)\]'
    replacement = fr'"float t_interval" [{float_val}]'
    updated_content = re.sub(pattern, replacement, file_content)
    with open(sys.argv[1], 'w') as file:
        file.write(updated_content)

if __name__ == "__main__":
    double_ablation_experiments()
    # convergence_experiments()
    # scattering_experiments()
    # gatewidth_experiments()
