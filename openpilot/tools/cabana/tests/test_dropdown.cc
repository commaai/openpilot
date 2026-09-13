#include <cassert>

#include "tools/cabana/ui/dropdown.h"

// Run without a display or renderer. Exercise popup lifetime and parent spacing,
// including nested menus: mismatched style stacks also trigger ImGui assertions.
int main() {
  ImGui::CreateContext();
  auto &io = ImGui::GetIO();
  io.IniFilename = nullptr;
  io.DisplaySize = ImVec2(800, 600);
  io.DeltaTime = 1.0f / 60.0f;
  unsigned char *pixels;
  int width, height;
  io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
  float combo_row_height = 0;
  for (int frame = 0; frame < 4; ++frame) {
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(io.DisplaySize);
    ImGui::Begin("Parent");
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(2, 0));
    const int stack_size = GImGui->StyleVarStack.Size;
    const bool closed_popup = dropdown::BeginPopup("closed");
    assert(!closed_popup);
    if (closed_popup) dropdown::EndPopup();
    assert(GImGui->StyleVarStack.Size == stack_size);
    int index = 0;
    dropdown::Combo("closed combo", &index, "One\0Two\0");
    assert(ImGui::GetStyle().ItemSpacing.y == 0);
    const ImGuiID combo_id = ImHashStr("##ComboPopup", 0, ImGui::GetID("open combo"));
    ImGui::OpenPopupEx(combo_id);
    const bool combo_open = dropdown::BeginCombo("open combo", "One");
    assert(combo_open);
    if (combo_open) {
      assert(ImGui::GetStyle().ItemSpacing.y == dropdown::SPACING_Y);
      dropdown::Item("One", nullptr, true);
      combo_row_height = ImGui::GetItemRectSize().y;
      const float selected_width = ImGui::GetItemRectSize().x;
      dropdown::Item("Two");
      assert(ImGui::GetItemRectSize().y == combo_row_height);
      assert(ImGui::GetItemRectSize().x == selected_width);
      ImGui::CloseCurrentPopup();
      dropdown::EndCombo();
    }
    assert(GImGui->StyleVarStack.Size == stack_size);
    ImGui::OpenPopup("popup");
    if (dropdown::BeginPopup("popup")) {
      assert(ImGui::GetStyle().ItemSpacing.y == dropdown::SPACING_Y);
      assert(ImGui::GetCurrentWindow()->WindowPadding.y == dropdown::PADDING_Y);
      dropdown::Item("An action");
      assert(ImGui::GetItemRectSize().y == combo_row_height);
      dropdown::Item("Checked", nullptr, true);
      assert(ImGui::GetItemRectSize().y == combo_row_height);
      bool checked = true;
      dropdown::Item("Disabled toggle", nullptr, &checked, false);
      assert(checked);
      assert(ImGui::GetItemRectSize().y == combo_row_height);
      ImGui::OpenPopup("Nested");
      if (dropdown::BeginMenu("Nested")) {
        dropdown::Item("Nested action");
        assert(ImGui::GetItemRectSize().y == combo_row_height);
        dropdown::EndMenu();
      }
      dropdown::EndPopup();
    }
    assert(GImGui->StyleVarStack.Size == stack_size);
    assert(ImGui::GetStyle().ItemSpacing.y == 0);
    ImGui::PopStyleVar();
    ImGui::End();
    ImGui::Render();
  }
  ImGui::DestroyContext();
}
