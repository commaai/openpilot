#!/usr/bin/env python
import pygame  # pylint: disable=import-error

# Define some colors
BLACK = (   0,   0,   0)
WHITE = ( 255, 255, 255)

# This is a simple class that will help us print to the screen
# It has nothing to do with the joysticks, just outputting the
# information.
class TextPrint:
    def __init__(self):
        self.reset()
        self.font = pygame.font.Font(None, 20)

    def printf(self, screen, textString):
        textBitmap = self.font.render(textString, True, BLACK)
        screen.blit(textBitmap, [self.x, self.y])
        self.y += self.line_height

    def reset(self):
        self.x = 10
        self.y = 10
        self.line_height = 15

    def indent(self):
        self.x += 10

    def unindent(self):
        self.x -= 10


pygame.init()

# Set the width and height of the screen [width,height]
size = [500, 700]
screen = pygame.display.set_mode(size)

pygame.display.set_caption("My Game")

#Loop until the user clicks the close button.
done = False

# Used to manage how fast the screen updates
clock = pygame.time.Clock()

# Initialize the joysticks
pygame.joystick.init()

# Get ready to print
textPrint = TextPrint()

# -------- Main Program Loop -----------
while not done:
    # EVENT PROCESSING STEP
    for event in pygame.event.get():  # User did something
        if event.type == pygame.QUIT:  # If user clicked close
            done = True  # Flag that we are done so we exit this loop

        # Possible joystick actions: JOYAXISMOTION JOYBALLMOTION JOYBUTTONDOWN JOYBUTTONUP JOYHATMOTION
        if event.type == pygame.JOYBUTTONDOWN:
            print("Joystick button pressed.")
        if event.type == pygame.JOYBUTTONUP:
            print("Joystick button released.")

    # DRAWING STEP
    # First, clear the screen to white. Don't put other drawing commands
    # above this, or they will be erased with this command.
    screen.fill(WHITE)
    textPrint.reset()

    # Get count of joysticks
    joystick_count = pygame.joystick.get_count()

    textPrint.printf(screen, "Number of joysticks: {}".format(joystick_count) )
    textPrint.indent()

    # For each joystick:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    textPrint.printf(screen, "Joystick {}".format(0) )
    textPrint.indent()

    # Get the name from the OS for the controller/joystick
    name = joystick.get_name()
    textPrint.printf(screen, "Joystick name: {}".format(name) )

    # Usually axis run in pairs, up/down for one, and left/right for
    # the other.
    axes = joystick.get_numaxes()
    textPrint.printf(screen, "Number of axes: {}".format(axes) )
    textPrint.indent()

    for i in range( axes ):
        axis = joystick.get_axis( i )
        textPrint.printf(screen, "Axis {} value: {:>6.3f}".format(i, axis) )
    textPrint.unindent()

    buttons = joystick.get_numbuttons()
    textPrint.printf(screen, "Number of buttons: {}".format(buttons) )
    textPrint.indent()

    for i in range( buttons ):
        button = joystick.get_button( i )
        textPrint.printf(screen, "Button {:>2} value: {}".format(i, button) )
    textPrint.unindent()

    textPrint.unindent()

    # ALL CODE TO DRAW SHOULD GO ABOVE THIS COMMENT

    # Go ahead and update the screen with what we've drawn.
    pygame.display.flip()

    # Limit to 20 frames per second
    clock.tick(20)

# Close the window and quit.
# If you forget this line, the program will 'hang'
# on exit if running from IDLE.
pygame.quit()
