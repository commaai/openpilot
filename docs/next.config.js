const withNextra = require('nextra')({
  theme: 'nextra-theme-docs',
  themeConfig: './theme.config.jsx'
})

const isProd = process.env.NODE_ENV === 'production'
module.exports = withNextra({
  output: 'export',
  assetPrefix: isProd ? process.env.ASSET_PREFIX : undefined,
  basePath: isProd? process.env.BASE_PATH : undefined,
  images: {
      unoptimized: true
  }
})
