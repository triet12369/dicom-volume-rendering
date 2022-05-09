def to_rgb_points(colormap):
    rgb_points = []
    for item in colormap:
        crange = item['range']
        color = item['color']
        for idx, r in enumerate(crange):
            if len(color) == len(crange):
                rgb_points.append([r] + color[idx])
            else:
                rgb_points.append([r] + color[0])

    return rgb_points