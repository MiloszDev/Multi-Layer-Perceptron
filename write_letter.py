import pygame
import sys

pygame.init()

width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Drawing App")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def save_drawing(filename="drawing.png"):
    pygame.image.save(screen, filename)

def main():
    screen.fill(WHITE)

    drawing = False
    last_pos = None
    brush_size = 5
    brush_color = BLACK

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
                last_pos = event.pos

            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    save_drawing("my_drawing.png")
                    print("Drawing saved as 'my_drawing.png'.")

        if drawing:
            mouse_pos = pygame.mouse.get_pos()
            if last_pos:
                pygame.draw.line(screen, brush_color, last_pos, mouse_pos, brush_size)
            last_pos = mouse_pos

        pygame.display.update()

if __name__ == "__main__":
    main()
