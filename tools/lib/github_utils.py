import base64
import requests
from http import HTTPMethod

class GithubUtils:
  def __init__(self, token, owner='commaai', repo='openpilot', container='ci-artifacts'):
    self.OWNER = owner
    self.REPO = repo
    self.TOKEN = token
    self.CONTAINER = container

  @property
  def API_ROUTE(self):
    return f"https://api.github.com/repos/{self.OWNER}/{self.REPO}"

  def api_call(self, path, data="", method=HTTPMethod.GET, accept=""):
    headers = {"Authorization": f"Bearer {self.TOKEN}", \
               "Accept": f"application/vnd.github{accept}+json"}
    path = self.API_ROUTE + f'/{path}'
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
      data = f'{{"message":"uploading {file_name}", \
                    "branch":"{bucket}", \
                    "commiter":{{"name":"Vehicle Researcher", "email": "user@comma.ai"}}, \
                    "content":"{encoded}"}}'
      github_path = f"contents/{file_name}"
      return self.api_call(github_path, data=data, method=HTTPMethod.PUT).ok

  def upload_files(self, bucket, files):
    return all(self.upload_file(bucket, path, file_name) for path,file_name in files)

  def get_file_url(self, bucket, file_name):
    github_path = f"contents/{file_name}?ref={bucket}"
    r = self.api_call(github_path)
    return r.json()['download_url'] if r.ok else None

  def get_pr_number(self, pr_branch):
    github_path = f"commits/{pr_branch}/pulls"
    r = self.api_call(github_path)
    return r.json()[0]['number'] if r.ok else None

  def comment_on_pr(self, comment, commenter, pr_branch):
    pr_number = self.get_pr_number(pr_branch)
    data = f'{{"body": "{comment}"}}'
    github_path = f'issues/{pr_number}/comments'
    r = self.api_call(github_path)
    if not r.ok:
      return False
    comments = [x['id'] for x in r.json() if x['user']['login'] == commenter]
    if comments:
      github_path = f'issues/comments/{comments[0]}'
      return self.api_call(github_path, data=data, method=HTTPMethod.PATCH).ok
    else:
      github_path=f'issues/{pr_number}/comments'
      return self.api_call(github_path, data=data, method=HTTPMethod.POST).ok

  def comment_images_on_pr(self, title, commenter, pr_branch, bucket, images):
    table = [f'<details><summary>{title}</summary><table>']
    for i,f in enumerate(images):
      if not (i % 2):
        table.append('<tr>')
      table.append(f'<td><img src=\\"https://raw.githubusercontent.com/{self.OWNER}/{self.CONTAINER}/{bucket}/{f}\\"></td>')
      if (i % 2):
        table.append('</tr>')
    table.append('</table></details>')
    table = ''.join(table)

    return self.comment_on_pr(table, commenter, pr_branch)
