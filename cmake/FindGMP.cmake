message( STATUS "Looking for GMP")
find_library(LIBGMP gmp PATHS /usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64)
if( LIBGMP STREQUAL LIBGMP-NOTFOUND)
   set(${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE "GMP library was not found")
   set(${CMAKE_FIND_PACKAGE}-FOUND FALSE)
else()
   message( STATUS "Found libGMP: ${LIBGMP}")
   find_file(GMP_H "gmp.h" PATHS /usr/include /usr/local/include /opt/local/include ${GMP_DIR}/include)
   if(GMP_H_NOTFOUND)
      set(${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE "gmp.h header file not found")
      set(${CMAKE_FIND_PACAKGE_NAME}_FOUND FALSE)
   else()
      message( STATUS "Found gmp.h: ${GMP_H}")
      get_filename_component(GMP_INCLUDE_DIR ${GMP_H} DIRECTORY CACHE)
      add_library(gmp UNKNOWN IMPORTED)
      set_target_properties(gmp 
       						PROPERTIES
   						    IMPORTED_LOCATION ${LIBGMP}
   	  )
      target_include_directories(gmp INTERFACE ${GMP_INCLUDE_DIR})
      add_library(GMP::gmp ALIAS gmp)
      set(${CMAKE_FIND_PACKAGE_NAME}_FOUND TRUE)
   endif()
endif()