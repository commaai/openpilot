const withNextra = require('nextra')({
  theme: 'nextra-theme-docs',
  themeConfig: './theme.config.jsx'
})

module.exports = withNextra({
  output: 'standalone',
  images: {
    unoptimized: true,
  },
  // Conditionally set basePath based on ENV
  ...(process.env.GITHUB_REPOSITORY
    ? {
      basePath: `/${process.env.GITHUB_REPOSITORY.split('/')[1]}`
    }
    : {}),
})
