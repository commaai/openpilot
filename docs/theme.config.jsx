// TODO make this externally configurable for forks?
const gitRepo = 'https://github.com/commaai/openpilot'

export default {
  logo: (
    <>
      {/* TODO  */}
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
  }
}
