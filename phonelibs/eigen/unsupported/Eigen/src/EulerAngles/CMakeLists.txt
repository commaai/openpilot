FILE(GLOB Eigen_EulerAngles_SRCS "*.h")

INSTALL(FILES
  ${Eigen_EulerAngles_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/unsupported/Eigen/src/EulerAngles COMPONENT Devel
  )
