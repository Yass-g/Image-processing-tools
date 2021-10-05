import numpy as np
import matplotlib.pyplot as plt
import sys
from PIL import Image
from matplotlib import backend_bases
from pynput.keyboard import Key, Listener

# The goal of this script is to be able to open an image, to let the user click a certain amount of times to draw
# polygonal selections, and then write up those selections in a txt file.

# Loading images, creating the graph , initializing the number of permanently selected points, initializing the colours.
Im_Name = 'Gorille.jpg'
Im = Image.open(Im_Name)
fig = plt.figure()
selections = 0
colours = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white']

# Storing the points.
# selectedx and selectedy will contain arrays that correspond to the selections.
selectedx = []
selectedy = []
# pointsx and pointsy will store the current selection.
pointsx = []
pointsy = []

# Displaying the image, with the selected points, and the current selection.
def update() :

    global selectedx
    global selectedy
    global pointsy
    global pointsx
    global fig

    fig.clear()

    ax = fig.add_subplot(111)
    ax.imshow(Im)

    for k in range(selections) :
        plt.scatter(selectedx[k], selectedy[k], c = colours[k%8])

    plt.scatter(pointsx, pointsy, c = colours[(selections)%8])

update()

# This allows you to add points in the current selection.
def select_points(event):
    
    print(plt.get_current_fig_manager().toolbar.mode)
    if plt.get_current_fig_manager().toolbar.mode != backend_bases._Mode.NONE:
        return

    global ix, iy
    ix, iy = event.xdata, event.ydata

    pointsx.append(int(ix))
    pointsy.append( int(iy))

    update()
    plt.show()

# This allows you to delete points in the current selections
def delete_points(key):

    global pointsx
    global pointsy

    sys.stdout.flush()
    if key.key == 'c' :

        pointsx = pointsx[:-1]
        pointsy = pointsy[:-1]

        fig.clear()
        update()
        plt.show()

# This allows you to save the current selection, and re-initialize the selection process.
def new_selection(key):

    global pointsx
    global pointsy
    global selectedx
    global selectedy
    global selections

    sys.stdout.flush()
    if key.key == 'n' :

        xcopy = pointsx[:]
        ycopy = pointsy[:]
        selectedx.append(xcopy)
        selectedy.append(ycopy)
        selections += 1

        pointsx = []
        pointsy = []

#This allows you to delete all the current selection, and, if it is already empty, delete the last one.
def delete_selection(key):

    global selectedx
    global selectedy
    global pointsy
    global pointsx
    global selections

    sys.stdout.flush()
    if key.key == 'm' :

        if len(pointsx) == 0 :
            if selections == 0 :
                return
            else :
                selectedx = selectedx[:-1]
                selectedy = selectedy[:-1]
                selections -= 1

        else :
            pointsx = []
            pointsy = []

        update()
        plt.show()

# This terminates the program.
def finish(key) :

    global selectedx
    global selectedy
    global pointsy
    global pointsx
    global selections

    if key.key == 'f' :

        # Ending the current selection.
        if(len(pointsx) != 0) :
            xcopy = pointsx[:]
            ycopy = pointsy[:]
            selectedx.append(xcopy)
            selectedy.append(ycopy)
            selections += 1

            pointsx = []
            pointsy = []

        # Writing in the file.
        f = open(f'{Im_Name}_Selections.txt', 'w')

        for i in range(selections) :

            f.write("Selection " + str(i) +"\n")

            for j in range(len(selectedx[i])) :

                f.write(str(selectedx[i][j]) + " " + str(selectedy[i][j]) + "\n")

        plt.close(fig)





fig.canvas.mpl_connect('button_press_event', select_points)
fig.canvas.mpl_connect('key_press_event', delete_points)
fig.canvas.mpl_connect('key_press_event', new_selection)
fig.canvas.mpl_connect('key_press_event', delete_selection)
fig.canvas.mpl_connect('key_press_event', finish)

plt.show()
