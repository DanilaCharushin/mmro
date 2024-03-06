import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from practice2.algo import Approximation

mpl.use("MacOSX")  # good fix, works only on MacOSX

num_points = 20
coords_x = np.random.randint(0, 50, num_points)
coords_y = np.random.randint(0, 50, num_points)

xm = np.average(coords_x)
ym = np.average(coords_y)
y = ((coords_x - xm) ** 2 + (coords_y - ym) ** 2) ** 0.5
print(list(y))

# y = [32.06243908, 11.80677771, 22.14497686, 8.48528137, 6.72309453, 28.14604768, 23.66854453, 4.12310563, 15.64608577, 22.64508777]
# y = [22.525208101147477, 21.692971211892576, 21.951879190629672, 26.13015499379979, 14.952758942750332, 17.843346098756253, 0.7648529270389174, 5.838236034968098, 14.514992249395107, 1.258967831201417, 26.774708215030092, 31.33823543213625, 15.520470353697403, 28.849350079334542, 31.088341866365273, 15.397564742516913, 22.18073488412861, 28.214978291680467, 33.31943877078364, 30.70480418436177]
y = [16.96179530592207, 17.59836640145897, 4.658594208556913, 11.2828409542987, 16.00632687408326, 20.528090510322677, 11.269538588602463, 30.912820964771235, 10.573670129146265, 23.927024470251208, 8.018883962248113, 6.848539990392114, 7.470107094279172, 12.12858194514099, 17.32346674312044, 22.345077757752374, 21.55, 25.278498768716467, 17.663592499828567, 18.791021792334764]

n = 7

x = [index for index, _ in enumerate(y)]

params = {
    "x": x,
    "y": y,
    "n": n,
}

print(y)

plt.plot(x, y, label="y")
plt.plot(x, y, ".", label="y (points)", markersize=10)

approx_func = Approximation.approximate_least_squares(**params)

y_approx = []
x_approx = []
x0 = 0
dx = 0.1

while x0 < x[-1]:
    x_approx.append(x0)
    y_approx.append(approx_func(x0))
    x0 += dx

plt.plot(x_approx, y_approx, label="y_approx")
plt.legend()
plt.show()
