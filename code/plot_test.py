from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# Some 2D geo arrays to plot (time,lat,lon)
data = np.random.random_sample((20,90,360))
lat = np.arange(len(data[0,:,0]))
lon = np.arange(len(data[0,0,:]))
lons,lats = np.meshgrid(lon,lat)

mode = 'imshow'
# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are animating three artists, the contour and 2 
# annotatons (title), in each frame
ims = []
for i in range(len(data[:,0,0])):
    if mode == 'contour':
        im = ax.contourf(lons,lats,data[i,:,:])
        add_arts = im.collections
    elif mode == 'imshow':
        im = ax.imshow(data[i, :, :], extent=[np.min(lons), np.max(lons),
                                               np.min(lats), np.max(lats)],
                       aspect='auto')
        add_arts = [im, ]

    text = 'title={0!r}'.format(i)
    te = ax.text(90, 90, text)
    an = ax.annotate(text, xy=(0.45, 1.05), xycoords='axes fraction')
    ims.append(add_arts + [te,an])

#ani = animation.ArtistAnimation(fig, ims, interval=70,repeat_delay=1000, blit=False)
ani = animation.ArtistAnimation(fig, ims)
## For saving the animation we need ffmpeg binary:
#FFwriter = animation.FFMpegWriter()
#ani.save('basic_animation.mp4', writer = FFwriter)
plt.show()