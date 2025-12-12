import webcolors

def closest_color(requested_rgb):
    min_colors = {}
    for name in webcolors.names("html4"):
        r, g, b = webcolors.name_to_rgb(name)
        min_colors[
            (r - requested_rgb[0]) ** 2
            + (g - requested_rgb[1]) ** 2
            + (b - requested_rgb[2]) ** 2
        ] = name
    return min_colors[min(min_colors.keys())]

def get_color_name(rgb_vector):
    rgb_255 = (
        int(rgb_vector[0] * 255),
        int(rgb_vector[1] * 255),
        int(rgb_vector[2] * 255),
    )
    try:
        return webcolors.rgb_to_name(rgb_255)
    except ValueError:
        return closest_color(rgb_255)