// TODO make this externally configurable for forks?
const gitRepo = 'https://github.com/commaai/openpilot'

export default {
  // TODO: SEO optimization ?
  // https://nextra.site/docs/docs-theme/theme-configuration#seo-options
  useNextSeoProps() {
    return {
      // title: 'openpilot',
      titleTemplate: 'openpilot - %s',
      defaultTitle: 'openpilot',
      description: 'open source driver assistance system that runs on 275+ car models',
      canonical: 'docs.comma.ai',
      twitter: {
        cardType: 'summary_large_image',
        handle: '@comma_ai'
      },
      opengraph: {
        type: 'website',
        locale: 'en_US',
        url: 'https://docs.comma.ai',
        title: 'openpilot',
        description: 'open source driver assistance system that runs on 275+ car models',
        images: [
          {
            url: 'https://github.com/commaai/openpilot/assets/8762862/f09e6d29-db2d-4179-80c2-51e8d92bdb5c',
            width: 1200,
            height: 630,
            alt: 'openpilot github'
          }
        ]
      }
    }
  },
  head: (
    <>
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      {/* favicon - github openpilot logo  - extract from git lfs storage */}
      <link rel="icon" type="image/png" href="https://media.githubusercontent.com/media/commaai/openpilot/master/selfdrive/assets/offroad/icon_openpilot.png" />
      {/* alt favicon - comma.ai logo */}
      {/* <link rel="icon" type="image/png" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAADzSURBVHgB7ZZbEcIwEEUvGAAcRAIS6gAJ1EklgAOKAkBBcVAcdFBQHJTNFGaSfLS7zCb85Mycrz528+hNgUxGxpqsyIYcPrbknjSIjCE7p3CovbZFRKaKf+1jNVEyirtLok4raMC6gzKD0CuUkTbQQxnpEgzcFy+Z990gx0ARG0AdZDNgoIyZaKLHj0uwgJwSY9isyAd5Jgvy4tzzIjdISAF/9A33Qe4mnCOMX/amjdWAehDN4W5O9vRrUcJf/6hHckiYDyckpob/Q7JGQg5BcYNE2FE2/ypewY/dIxJOew3/UyuggOQssFn/xBgyd2QySrwBMPGV3eNmgPcAAAAASUVORK5CYII=" sizes="32x32" /> */}
      {/* analytics */}
      <script type="text/javascript" defer="" src="https://plausible.io/js/script.outbound-links.js" data-domain="docs.comma.ai"></script>
      <script defer="" src="https://static.hotjar.com/c/hotjar-3254988.js?sv=6"></script>
    </>
  ),
  logo: (
    <>
      <img
        // src="https://github.com/commaai.png?size=48"
        src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAADzSURBVHgB7ZZbEcIwEEUvGAAcRAIS6gAJ1EklgAOKAkBBcVAcdFBQHJTNFGaSfLS7zCb85Mycrz528+hNgUxGxpqsyIYcPrbknjSIjCE7p3CovbZFRKaKf+1jNVEyirtLok4raMC6gzKD0CuUkTbQQxnpEgzcFy+Z990gx0ARG0AdZDNgoIyZaKLHj0uwgJwSY9isyAd5Jgvy4tzzIjdISAF/9A33Qe4mnCOMX/amjdWAehDN4W5O9vRrUcJf/6hHckiYDyckpob/Q7JGQg5BcYNE2FE2/ypewY/dIxJOew3/UyuggOQssFn/xBgyd2QySrwBMPGV3eNmgPcAAAAASUVORK5CYII="
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
  primaryHue: 115,
  primarySaturation: {
    light: 81, // tad easier to read on a white background
    dark: 87.2
  },
  docsRepositoryBase: gitRepo + '/blob/master/docs',
  // TODO add x-twitter
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
