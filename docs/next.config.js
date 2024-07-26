const withNextra = require('nextra')({
  theme: 'nextra-theme-docs',
  themeConfig: './theme.config.jsx'
})

const isProd = process.env.NODE_ENV === 'production'
module.exports = withNextra({
  output: 'standalone',
  images: {
    unoptimized: true,
  },
  assetPrefix: isProd ? process.env.ASSET_PREFIX : undefined,
  basePath: isProd? process.env.BASE_PATH : undefined,
  // Conditionally set basePath based on ENV
  ...(process.env.GITHUB_REPOSITORY
    ? {
      basePath: `/${process.env.GITHUB_REPOSITORY.split('/')[1]}`
    }
    : {}),
})
