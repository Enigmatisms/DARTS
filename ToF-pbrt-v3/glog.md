# googletest compilation

There might be error during external lib compilation (linking error):

`glog` typeinfo undefined, which might be difficult to solve. Therefore it happens
we can comment a few lines in the `CMakeLists.txt` in ext/src/glog:

```cmake
# comment the following to prevent glog from using googletest
find_package (GTest NO_MODULE)

if (GTest_FOUND)
  set (HAVE_LIB_GTEST 1)
endif (GTest_FOUND)
```

