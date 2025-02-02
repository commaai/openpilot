import sys
import select
import raylib

PROGRESS_BAR_WIDTH = 1000
PROGRESS_BAR_HEIGHT = 1000
ROTATION_RATE = 12.0
MARGIN = 200
TEXTURE_SIZE = 360
FONT_SIZE = 80

def main():
    raylib.InitWindow(PROGRESS_BAR_WIDTH, PROGRESS_BAR_HEIGHT, b"spinner")
    raylib.SetTargetFPS(30)

    comma_texture = raylib.LoadTexture(b"./selfdrive/assets/img_spinner_comma.png")
    if comma_texture.id == 0:
        print("Error loading comma texture")
        return

    spinner_texture = raylib.LoadTexture(b"./selfdrive/assets/img_spinner_track.png")
    if spinner_texture.id == 0:
        print("Error loading spinner texture")
        return

    rotation = 0.0
    user_input = ""

    while not raylib.WindowShouldClose():
        raylib.BeginDrawing()
        raylib.ClearBackground(raylib.BLACK)

        rotation = (rotation + ROTATION_RATE) % 360.0
        center = (raylib.GetScreenWidth() / 2, raylib.GetScreenHeight() / 2)

        comma_rec = (0, 0, comma_texture.width, comma_texture.height)
        spinner_rec = (0, 0, spinner_texture.width, spinner_texture.height)
        spinner_origin = (spinner_rec[2] / 2, spinner_rec[3] / 2)
        comma_position = (center[0] - comma_rec[2] / 2, center[1] - comma_rec[3] / 2)

        raylib.DrawTexturePro(spinner_texture, spinner_rec,
                          (center[0], center[1], TEXTURE_SIZE, TEXTURE_SIZE),
                          spinner_origin, rotation, raylib.WHITE)
        raylib.DrawTextureV(comma_texture, comma_position, raylib.WHITE)

        if select.select([sys.stdin], [], [], 0)[0]:
            user_input = sys.stdin.readline().strip()

        if user_input:
            y_pos = raylib.GetScreenHeight() - MARGIN - PROGRESS_BAR_HEIGHT
            if user_input.isdigit():
                bar = (center[0] - PROGRESS_BAR_WIDTH / 2, y_pos, PROGRESS_BAR_WIDTH, PROGRESS_BAR_HEIGHT)
                raylib.DrawRectangleRounded(bar, 0.5, 10, raylib.GRAY)

                progress = max(0, min(100, int(user_input)))
                bar = (bar[0], bar[1], bar[2] * progress / 100, bar[3])
                raylib.DrawRectangleRounded(bar, 0.5, 10, raylib.RAYWHITE)
            else:
                text_size = raylib.MeasureTextEx(raylib.GetFontDefault(), user_input.encode(), FONT_SIZE, 1.0)
                raylib.DrawTextEx(raylib.GetFontDefault(), user_input.encode(), (center[0] - text_size.x / 2, y_pos), FONT_SIZE, 1.0, raylib.WHITE)

        raylib.EndDrawing()

    raylib.UnloadTexture(comma_texture)
    raylib.UnloadTexture(spinner_texture)
    raylib.CloseWindow()

if __name__ == "__main__":
    main()

