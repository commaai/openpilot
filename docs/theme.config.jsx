// TODO make this externally configurable for forks?
const gitRepo = 'https://github.com/commaai/openpilot'

export default {
  // TODO: SEO optimization ?
  // https://nextra.site/docs/docs-theme/theme-configuration#seo-options
  useNextSeoProps() {
    return {
      titleTemplate: 'openpilot – make driving chill'
    }
  },
  head: (
    <>
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <meta property="og:title" content="openpilot" />
      <meta property="og:description" content="make driving chill." />
    </>
  ),
  logo: (
    <>
      <img
        src="https://github.com/commaai.png?size=48"
        alt="comma.ai logo"
        width="24"
        height="24"
        style={{ borderRadius: '50%' }}
      />
      <span style={{ marginLeft: '.4em', fontWeight: 800 }}>
        openpilot
      </span>
    </>
  ),
  project: {
    link: gitRepo
  },
  docsRepositoryBase: gitRepo + '/blob/master/docs',
  chat: {
    link: 'https://discord.comma.ai/'
  },
  sidebar: {
    defaultMenuCollapseLevel: 1
  },
  footer: {
    component: null
  }
}
