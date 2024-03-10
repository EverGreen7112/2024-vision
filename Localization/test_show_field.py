import math

import show_on_field


def main():
    show_on_field.thread.start()
    x = 0
    y = 0
    rotation = 0
    show_on_field.run_thread = True
    while True:
        show_on_field.xyz = [x, 0, y]
        show_on_field.rotation = math.radians(rotation)


if __name__ == '__main__':
    main()
