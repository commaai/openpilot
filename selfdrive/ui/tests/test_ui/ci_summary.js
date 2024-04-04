const nodeFs = require("node:fs");
const nodePath = require("node:path");
const nodeChildProcess = require("node:child_process");
const knownLanguages = JSON.parse(nodeFs.readFileSync(
  nodePath.resolve(
    __dirname,
    "../../translations/languages.json"
  ), "utf8")
);

const REPORT_DIR = nodePath.resolve(__dirname, "report/screenshots");

function getImageLinkFromPath(runEnv, language, name, path) {
  const cmd = `${runEnv} python ${nodePath.resolve(__dirname, "upload_image_to_azure.py")} ${path} ${language}-${name}`;
  console.log(`uploading with ${cmd}`)
  let link = nodeChildProcess.execSync(cmd, {
    encoding: "utf-8",
  });

  return `![${name}](${link})`;
}

module.exports = async ({ core, glob, runEnv }) => {
  let summary = await core.summary
    .addHeading('UI Screenshots');

  let languages = nodeFs.readdirSync(REPORT_DIR).map((x) => nodePath.basename(x));

  for (let language_code of languages) {
    const language_name = Object.keys(knownLanguages).find(key => knownLanguages[key] === language_code);
    const language_dir = nodePath.resolve(REPORT_DIR, language_code);
    summary = summary.addHeading(language_name, '2');

    let table = [
      [{ data: 'Case', header: true }, { data: 'Screenshot', header: true }],
    ];

    const globber = await glob.create(language_dir + "/**/*");
    const files = await globber.glob();

    for (let file of files) {
      let name = file.split(language_dir + '/')[1].split(".png")[0];

      table.push([name, getImageLinkFromPath(runEnv, language_code, name, file)]);
    }

    summary = summary.addTable(table)
  }

  summary.write();
}