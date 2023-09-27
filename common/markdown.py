from typing import List

HTML_REPLACEMENTS = [
  (r'&', r'&amp;'),
  (r'"', r'&quot;'),
]


def parse_markdown(text: str, tab_length: int = 2) -> str:
  lines = text.split("\n")
  output: List[str] = []
  list_level = 0

  def end_outstanding_lists(level: int, end_level: int) -> int:
    while level > end_level:
      level -= 1
      output.append("</ul>")
      if level > 0:
        output.append("</li>")
    return end_level

  for i, line in enumerate(lines):
    if i + 1 < len(lines) and lines[i + 1].startswith("==="):  # heading
      output.append(f"<h1>{line}</h1>")
    elif line.startswith("==="):
      pass
    elif line.lstrip().startswith("* "):  # list
      line_level = 1 + line.count(" " * tab_length, 0, line.index("*"))
      if list_level >= line_level:
        list_level = end_outstanding_lists(list_level, line_level)
      else:
        list_level += 1
        if list_level > 1:
          output[-1] = output[-1].replace("</li>", "")
        output.append("<ul>")
      output.append(f"<li>{line.replace('*', '', 1).lstrip()}</li>")
    else:
      list_level = end_outstanding_lists(list_level, 0)
      if len(line) > 0:
        output.append(line)

  end_outstanding_lists(list_level, 0)
  output_str = "\n".join(output) + "\n"

  for (fr, to) in HTML_REPLACEMENTS:
    output_str = output_str.replace(fr, to)

  return output_str
