from packages import *


##############################
# Error decomposition ########
##############################
    

Decom = False
if Decom:
    G_Quadrath = [[ 2.628, 2.4308, 2.514,  3.3734, 4.33,  3.1,6.0416],
            [ 2.07, 2.2273, 1.976,  2.3504,  1.412, 2.223,1.7956],
            [ 0.0173, 0.022,  0.0252, 0.0877, 1.66, 0.39,0.8991]]

    Points = [ 
            [
                0.122,.2697,.3908,2.1744,1.9618, 1.9303,1.9201,2.0348
            ],
            [
                .3345,1.2617,1.0142,1.568,1.6497,1.4706,1.597,1.5909
                ],
            [
                .0149,.0635,.0753,.0117,.013,0.0099,.0083,.009
            ]]



    gq_columns = ('C=3.5, e=0.1', 'C=3.5, e=0.05', 'C=6, e=0.05', 'C=6, e=0.01', 'C=14, e=0.01', 'C=14, e=0.005','C=20, e=0.005')
    point_columns = ("5","6","7","8","9","25","100","1000")

    data = G_Quadrath
    columns = gq_columns
    rows = ["ZRL","|Du| ","W(u)"]

    values = np.arange(0, 8, 500)
    value_increment = 1
    plt.figure(figsize=(15, 12), dpi=200)
    # Get some pastel shades for the colors
    colors = plt.cm.Set2(np.linspace(0.0, 1.5, len(rows)))
    n_rows = len(data)

    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4

    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.zeros(len(columns))

    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(n_rows):
        plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
        y_offset = y_offset + data[row]
        cell_text.append(['%1.4f' % x  for x in data[row]])
    # Reverse colors and text labels to display the last value at the top.
    colors = colors[::-1]
    cell_text.reverse()

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                        rowLabels=rows,
                        rowColours=colors,
                        colLabels=columns,
                        loc='bottom')
    the_table.set_fontsize(26)
    cellDict = the_table.get_celld()
    for i in range(0,len(columns)):
        cellDict[(0,i)].set_height(.07)
        for j in range(1,len(rows)+1):
            cellDict[(j,i)].set_height(.07)
    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)

    plt.ylabel("Total loss")
    plt.yticks([1,2,3,4,5,6,7,8])
    plt.xticks([])
    plt.title('Composition of the loss')

    plt.show()
    
    

######################
# Accuracy of NN  ####
######################   
    
 
    
pol = False

if pol:
    
    x = [ 1,2,4,8,16,32,64,128,256,512]
    xx =[ 19, 37, 73, 145, 289,  577, 1153, 2305, 4609, 9217   ]
    y = [0.7911,0.4779,.3958,.4212,.184,.0363,.0250,.0131,.006,0.003]
    yy =[ 766,648,605,590,457,203,168,83,59,45 ]
    plt.plot(x,y)
    h = np.arange(1,512)
    plt.plot(h, 1/h)
    plt.show()
    plt.loglog(x,y)
    plt.plot(h, 1/h)
    plt.show()
    
asdf = True

if asdf:
    x = [ 1,2,4,8,16,32,64,128,256,512]
    xx =[ 19, 37, 73, 145, 289,  577, 1153, 2305, 4609, 9217   ]
    y = [0.7911,0.4779,.3958,.4212,.184,.0363,.0250,.0131,.006,0.003]
    yy =[ 766,648,605,590,457,203,168,83,59,45 ]
    h = np.arange(1,512)
    plt.xlabel("Number of Neurons")
    plt.ylabel("Loss")
    plt.loglog(x,y, 'p-', label="$\Psi(x)$")
    plt.plot(h, 1/h, label="$\\frac{1}{x}$")
    plt.legend(loc="upper right")

    plt.show()
    
    