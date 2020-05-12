set(Eigen_CXX11_HEADERS Tensor TensorSymmetry ThreadPool)

install(FILES
  ${Eigen_CXX11_HEADERS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/unsupported/Eigen/CXX11 COMPONENT Devel
  )

install(DIRECTORY src DESTINATION ${INCLUDE_INSTALL_DIR}/unsupported/Eigen/CXX11 COMPONENT Devel FILES_MATCHING PATTERN "*.h")
