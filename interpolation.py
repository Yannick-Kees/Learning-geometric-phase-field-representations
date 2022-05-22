from networks import *

f = ParkEtAl(2, [128]*3, [], geometric_init=True, FourierFeatures=False, num_features = 6, sigma = 2 )
f.load_state_dict(torch.load(r"C:\Users\Yannick\Desktop\MA\Programming part\models\circle.pth", map_location=device))


g = ParkEtAl(2, [128]*3, [], geometric_init=True, FourierFeatures=False, num_features = 6, sigma = 2 )
g.load_state_dict(torch.load(r"C:\Users\Yannick\Desktop\MA\Programming part\models\square.pth", map_location=device))


def Interpolate(f, g, t, i):
    color_plot_interpolate(f,g,t, i, True)
    draw_phase_field_interpolate(f, g, t, .5, .5, i, True)

for i in range(201):
    Interpolate(f, g, i/200, i)

