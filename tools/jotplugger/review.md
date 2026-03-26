# fix if statements with no brackets, always use brackets

for example:
        if (add_additional_source(&editor, static_cast<const char *>(payload->Data)))
          editor.preview_is_result = false;

# fix if statements on a single line that should be on separate lines

for example:

if (ImGui::Button("Close", ImVec2(120.0f, 0.0f))) { editor.open = false; editor.request_select = false; }

# 
