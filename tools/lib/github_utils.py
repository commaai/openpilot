import base64
import requests
from http import HTTPMethod

class GithubUtils:
  def __init__(self, api_token, data_token, owner='commaai', api_repo='openpilot', data_repo='ci-artifacts'):
    self.OWNER = owner
    self.API_REPO = api_repo
    self.DATA_REPO = data_repo
    self.API_TOKEN = api_token
    self.DATA_TOKEN = data_token

  @property
  def API_ROUTE(self):
    return f"https://api.github.com/repos/{self.OWNER}/{self.API_REPO}"

  @property
  def DATA_ROUTE(self):
    return f"https://api.github.com/repos/{self.OWNER}/{self.DATA_REPO}"

  def api_call(self, path, data="", method=HTTPMethod.GET, accept="", data_call=False):
    token = self.DATA_TOKEN if data_call else self.API_TOKEN
    if token:
      headers = {"Authorization": f"Bearer {self.DATA_TOKEN if data_call else self.API_TOKEN}", \
                 "Accept": f"application/vnd.github{accept}+json"}
    else:
      headers = {}
    path = f'{self.DATA_ROUTE if data_call else self.API_ROUTE}/{path}'
    match method:
      case HTTPMethod.GET:
        return requests.get(path, headers=headers)
      case HTTPMethod.PUT:
        return requests.put(path, headers=headers, data=data)
      case HTTPMethod.POST:
        return requests.post(path, headers=headers, data=data)
      case HTTPMethod.PATCH:
        return requests.patch(path, headers=headers, data=data)
      case _:
        raise NotImplementedError()

  def upload_file(self, bucket, path, file_name):
    with open(path, "rb") as f:
      encoded = base64.b64encode(f.read()).decode()

      # check if file already exists
      sha = self.get_file_sha(bucket, file_name)
      sha = f'"sha":"{sha}",' if sha else ''

      data = f'{{"message":"uploading {file_name}", \
                    "branch":"{bucket}", \
                    "committer":{{"name":"Vehicle Researcher", "email": "user@comma.ai"}}, \
                    {sha} \
                    "content":"{encoded}"}}'
      github_path = f"contents/{file_name}"
      if not self.api_call(github_path, data=data, method=HTTPMethod.PUT, data_call=True).ok:
        raise Exception(f"Error uploading {file_name} to {bucket}")

  def upload_files(self, bucket, files):
    all(self.upload_file(bucket, path, file_name) for file_name,path in files)

  def get_file_url(self, bucket, file_name):
    github_path = f"contents/{file_name}?ref={bucket}"
    r = self.api_call(github_path, data_call=True)
    return r.json()['download_url'] if r.ok else None

  def get_file_sha(self, bucket, file_name):
    github_path = f"contents/{file_name}?ref={bucket}"
    r = self.api_call(github_path, data_call=True)
    return r.json()['sha'] if r.ok else None

  def get_pr_number(self, pr_branch):
    github_path = f"commits/{pr_branch}/pulls"
    r = self.api_call(github_path)
    return r.json()[0]['number'] if r.ok else None

  def comment_on_pr(self, comment, commenter, pr_branch):
    pr_number = self.get_pr_number(pr_branch)
    if not pr_number:
      raise Exception(f"No PR found for branch {pr_branch}")
    data = f'{{"body": "{comment}"}}'
    github_path = f'issues/{pr_number}/comments'
    r = self.api_call(github_path)
    comments = [x['id'] for x in r.json() if x['user']['login'] == commenter]
    if comments:
      github_path = f'issues/comments/{comments[0]}'
      if not self.api_call(github_path, data=data, method=HTTPMethod.PATCH).ok:
        raise Exception(f"Can't edit {commenter} previous comment on PR#{pr_number}")
    else:
      github_path=f'issues/{pr_number}/comments'
      if not self.api_call(github_path, data=data, method=HTTPMethod.POST).ok:
        raise Exception(f"Can't post comment on PR#{pr_number}")

  # upload files to github and comment them on the pr
  def comment_images_on_pr(self, title, commenter, pr_branch, bucket, images):
    self.upload_files(bucket, images)
    table = [f'<details><summary>{title}</summary><table>']
    for i,f in enumerate(images):
      if not (i % 2):
        table.append('<tr>')
      table.append(f'<td><img src=\\"https://raw.githubusercontent.com/{self.OWNER}/{self.DATA_REPO}/{bucket}/{f[0]}\\"></td>')
      if (i % 2):
        table.append('</tr>')
    table.append('</table></details>')
    table = ''.join(table)
    self.comment_on_pr(table, commenter, pr_branch)
