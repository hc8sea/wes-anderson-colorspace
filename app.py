import requests
import shutil
import cv2
import palettable
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance
import plotly.express as px

st.set_page_config(
        page_title="Wes Anderson Color Palettes",
)

#This function generates an array of pixels from external picture

def url_to_pixel_matrix(url):
  filename = url.split("/")[-1]
  r = requests.get(url, stream = True)
  r.raw.decode_content = True    # Open a local file with wb ( write binary ) permission.
  with open(filename,'wb') as f:
          shutil.copyfileobj(r.raw, f)
  pixel_matrix_bgr = cv2.imread(filename) #original img
  pixel_matrix_rgb = cv2.cvtColor(pixel_matrix_bgr, cv2.COLOR_BGR2RGB)
  pixel_matrix_downscale = cv2.resize(pixel_matrix_rgb, (50,50)) #rescale image
  return pixel_matrix_downscale

#This function generates a pixel list from a pixel array

def pixel_matrix_to_pixel_flat(pixel_matrix):
  pixel_list=[]
  for i in pixel_matrix:
    for j in i:
        if int(j[0])+int(j[1])+int(j[2]) > 10:
            pixel_list.append(j)
  pixel_flat=np.array(pixel_list)
  return pixel_flat

def main():

    st.header('Color palettes in the RGB space')

    st.write(
"""
This dashboard intends to display the following:

    1) Image capture and manipulation via Streamlit e OpenCV
    2) K-Means Clustering
    3) Data wrangling via Pandas e Numpy
    4) 3D plots using Plotly

     """)

    st.write('repository link: https://github.com/hc8sea/wes-anderson-colorspace')

    img_file_buffer = st.camera_input("Take a picture and wait a moment")

    if img_file_buffer is not None:

        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        size_xy=cv2_img.shape[:2][::-1]
        pixel_matrix_downscale = cv2.resize(cv2_img, (50,50))
        pixel_matrix_rgb = cv2.cvtColor(pixel_matrix_downscale, cv2.COLOR_BGR2RGB)

        pixel_flat = pixel_matrix_to_pixel_flat(pixel_matrix_rgb)



        kmeans_ = KMeans(init = "k-means++", n_clusters = 13, n_init = 12).fit(pixel_flat)
        k_means_cluster_centers_ = kmeans_.cluster_centers_/255.


        color0_=k_means_cluster_centers_[0]
        color1_=k_means_cluster_centers_[1]
        color2_=k_means_cluster_centers_[2]
        color3_=k_means_cluster_centers_[3]
        color4_=k_means_cluster_centers_[4]
        color5_=k_means_cluster_centers_[5]
        color6_=k_means_cluster_centers_[6]
        color7_=k_means_cluster_centers_[7]
        color8_=k_means_cluster_centers_[8]
        color9_=k_means_cluster_centers_[9]
        color10_=k_means_cluster_centers_[10]
        color11_=k_means_cluster_centers_[11]
        color12_=k_means_cluster_centers_[12]

        #Setting the graph's parameters

        fig_, ax_ = plt.subplots(figsize=(10, 1))
        fig_.subplots_adjust(bottom=0.5)
        cmap_ = (mpl.colors.ListedColormap([color0_,color1_,color2_,color3_,color4_,color5_,color6_,color7_,color8_,color9_,color10_,color11_,color12_]))
        bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13]
        norm_ = mpl.colors.BoundaryNorm(bounds, cmap_.N)

        #Generating the graph

        fig_.colorbar(
            mpl.cm.ScalarMappable(cmap=cmap_, norm=norm_),
            cax=ax_,
            boundaries=[0] + bounds + [13],  # Adding values for extensions.
            extend='neither', #it was both
            ticks=bounds,
            spacing='proportional',
            orientation='horizontal',
            label='',
        )

        #Ploting the graph

        st.write('This is your color palette:')

        st.pyplot(fig_)

	#Now we are going to compare the color palette we just got with color palettes from wes anderson movies
        #Reference palettes list, for comparison

        colors=[
        [palettable.wesanderson.Aquatic1_5.colors, [0], [palettable.wesanderson.Aquatic1_5.number]],
        [palettable.wesanderson.Aquatic2_5.colors, [1], [palettable.wesanderson.Aquatic2_5.number]],
        [palettable.wesanderson.Aquatic3_5.colors, [2], [palettable.wesanderson.Aquatic3_5.number]],
        [palettable.wesanderson.Cavalcanti_5.colors, [3], [palettable.wesanderson.Cavalcanti_5.number]],
        [palettable.wesanderson.Darjeeling2_5.colors,[4], [palettable.wesanderson.Darjeeling2_5.number]],
        [palettable.wesanderson.Darjeeling3_5.colors, [5], [palettable.wesanderson.Darjeeling3_5.number]],
        [palettable.wesanderson.Darjeeling4_5.colors, [6], [palettable.wesanderson.Darjeeling4_5.number]],
        [palettable.wesanderson.FantasticFox1_5.colors, [7], [palettable.wesanderson.FantasticFox1_5.number]],
        [palettable.wesanderson.FantasticFox2_5.colors, [8], [palettable.wesanderson.FantasticFox2_5.number]],
        [palettable.wesanderson.GrandBudapest4_5.colors, [9], [palettable.wesanderson.GrandBudapest4_5.number]],
        [palettable.wesanderson.GrandBudapest5_5.colors, [10], [palettable.wesanderson.GrandBudapest5_5.number]],
        [palettable.wesanderson.IsleOfDogs1_5.colors, [11], [palettable.wesanderson.IsleOfDogs1_5.number]],
        [palettable.wesanderson.Moonrise1_5.colors, [12], [palettable.wesanderson.Moonrise1_5.number]],
        [palettable.wesanderson.Moonrise4_5.colors, [13], [palettable.wesanderson.Moonrise4_5.number]],
        [palettable.wesanderson.Moonrise6_5.colors, [14], [palettable.wesanderson.Moonrise6_5.number]],
        [palettable.wesanderson.Moonrise7_5.colors, [15], [palettable.wesanderson.Moonrise7_5.number]],
        [palettable.wesanderson.Royal2_5.colors, [16], [palettable.wesanderson.Royal2_5.number]],
        [palettable.wesanderson.Royal3_5.colors, [17], [palettable.wesanderson.Royal3_5.number]],
        [palettable.wesanderson.Zissou_5.colors, [18], [palettable.wesanderson.Zissou_5.number]],
        [palettable.wesanderson.Chevalier_4.colors, [19], [palettable.wesanderson.Chevalier_4.number]],
        [palettable.wesanderson.Darjeeling1_4.colors, [20], [palettable.wesanderson.Darjeeling1_4.number]],
        [palettable.wesanderson.GrandBudapest1_4.colors, [21], [palettable.wesanderson.GrandBudapest1_4.number]],
        [palettable.wesanderson.GrandBudapest2_4.colors, [22], [palettable.wesanderson.GrandBudapest2_4.number]],
        [palettable.wesanderson.IsleOfDogs3_4.colors, [23], [palettable.wesanderson.IsleOfDogs3_4.number]],
        [palettable.wesanderson.Mendl_4.colors, [24], [palettable.wesanderson.Mendl_4.number]],
        [palettable.wesanderson.Moonrise2_4.colors, [25], [palettable.wesanderson.Moonrise2_4.number]],
        [palettable.wesanderson.Moonrise3_4.colors, [26], [palettable.wesanderson.Moonrise3_4.number]],
        [palettable.wesanderson.Royal1_4.colors, [27], [palettable.wesanderson.Royal1_4.number]],
        [palettable.wesanderson.GrandBudapest3_6.colors, [28], [palettable.wesanderson.GrandBudapest3_6.number]],
        [palettable.wesanderson.IsleOfDogs2_6.colors, [29], [palettable.wesanderson.IsleOfDogs2_6.number]],
        [palettable.wesanderson.Moonrise5_6.colors, [30], [palettable.wesanderson.Moonrise5_6.number]],
        ]
        colors_=[]

        #Turning the previous list into a DataFrame

        for j in range(31):
          for i in range(colors[j][2][0]):
           colors_.append([colors[j][0][i][0]/255.,colors[j][0][i][1]/255.,colors[j][0][i][2]/255.,colors[j][1][0],colors[j][2][0]])
        colors_=np.array(colors_)

        df=pd.DataFrame(colors_,columns=('r','g','b','palette','number'))

        X=df[['r','g','b']]
        Y=df[['palette','number']]

        X=X.to_numpy(dtype= np.float)

	#Creating a 2D array where the rows represent the reference color's palette and the columns represent the colors from the detected color palette
	#The individual values of each cell are the euclidian distances - in the RGB colorspace - between the row's color and the columns'color


        teste = np.ndarray(shape=(len(X),len(k_means_cluster_centers_)+1))
        for i in range(len(X)):
            for j in range(len(k_means_cluster_centers_)):
                teste[i][j] = distance.euclidean(k_means_cluster_centers_[j],X[i])
            teste[i][-1] = Y['palette'][i]

	#Creating an array called gabarito to find out what reference color palette is closer to the detected color palette

        gab = pd.DataFrame(teste, columns=('color0_','color1_','color2_','color3_','color4_','color5_','color6_','color7_','color8_','color9_','color10_','color11_','color12_','palette'))

        gabarito = np.ndarray(shape=(31,3))
        for i in range(31):
            gabarito[i][0] = min(gab[gab['palette']==i].iloc[0][:-1])            #Encontra a menor distância euclidiana entre as cores
            gabarito[i][1] = gab[gab['palette']==i].iloc[0][:-1].mean()          #Calcula a média de todas as distâncias euclidianas
            gabarito[i][2] = gabarito[i][0] + gabarito[i][1]                     #Soma os dois parâmetros anteriores

        gabs = pd.DataFrame(gabarito, columns=('min','mean','combined'))

        #Out of three criteria, the option is to use the mean value of the distance

        wes_ = (gabs[gabs['mean'] == min(gabs['mean'])].index.values)[0]

	#The following section represents the match results along with a movie pic and a quote.

        st.write('According to your color palette you are in this particular Wes Anderson movie:')

        if wes_ == 0:
            colorfilm = palettable.wesanderson.Aquatic1_5.mpl_colors
            st.header('The Life Aquatic with Steve Zissou')
            st.write('"The deeper you go, the weirder life gets."')
            st.image('https://64.media.tumblr.com/8b7d9f3bb5eda62aa8a7ec46558be546/tumblr_nizhdliOFn1tvvqeko1_1280.jpg')
        elif wes_ == 1:
            colorfilm = palettable.wesanderson.Aquatic2_5.mpl_colors
            st.header('The Life Aquatic with Steve Zissou')
            st.write('"Please don’t make fun of me. I just wanted to flirt with you."')
            st.image('https://64.media.tumblr.com/26eb097655b1df589dc366836c352d82/tumblr_ns5gslLbgB1tvvqeko1_500.jpg')
        elif wes_ == 2:
            colorfilm = palettable.wesanderson.Aquatic3_5.mpl_colors
            st.header('The Life Aquatic with Steve Zissou')
            st.write('"We’ve never made great husbands, have we?"')
            st.image('https://64.media.tumblr.com/c4a18b28c87d80ed9833fba58ddde0f1/tumblr_o2p1rtcgsK1tvvqeko1_500.jpg')
        elif wes_ == 3:
            colorfilm = palettable.wesanderson.Cavalcanti_5.mpl_colors
            st.header('Castello Cavalcanti')
            st.write('"Castello Cavalcanti, how can I help?"')
            st.image('https://64.media.tumblr.com/90ff02882fb8e5c6302cddb13558969c/tumblr_n2bhkgazU91tvvqeko1_500.jpg')
        elif wes_ == 4:
            colorfilm = palettable.wesanderson.Darjeeling2_5.mpl_colors
            st.header('The Darjeeling Limited')
            st.write('"Fuck the itinerary."')
            st.image('https://64.media.tumblr.com/2815b755b493555dd4a74fc9f7c84bdb/tumblr_nj7cclt9qb1tvvqeko1_500.jpg')
        elif wes_ == 5:
            colorfilm = palettable.wesanderson.Darjeeling3_5.mpl_colors
            st.header('The Darjeeling Limited')
            st.write('"Welcome aboard."')
            st.image('https://64.media.tumblr.com/300310dd5e503f51b0b875e05db79791/tumblr_o6r7oiHYiI1tvvqeko1_500.jpg')
        elif wes_ == 6:
            colorfilm = palettable.wesanderson.Darjeeling4_5.mpl_colors
            st.header('The Darjeeling Limited')
            st.write('"I wonder if the three of us would’ve been friends in real life. Not as brothers, but as people."')
            st.image('https://64.media.tumblr.com/0c06d4a5d62da4e28671cb86c781d159/tumblr_ophelretIo1tvvqeko1_500.jpg')
        elif wes_ == 7:
            colorfilm = palettable.wesanderson.FantasticFox1_5.mpl_colors
            st.header('Fantastic Mr. Fox')
            st.write('"Mrs Fox: You know, you really are… fantastic."')
            st.write('"Mr Fox: I try."')
            st.image('https://64.media.tumblr.com/97284768c2a7255d394048e847480f4a/tumblr_n2q5vtQtS81tvvqeko1_500.jpg')
        elif wes_ == 8:
            colorfilm = palettable.wesanderson.FantasticFox2_5.mpl_colors
            st.header('Fantastic Mr. Fox')
            st.write('"Should we dance?"')
            st.image('https://64.media.tumblr.com/80f701e4c2a15b3a914b06519253f50f/tumblr_njlwde8PnX1tvvqeko1_500.jpg')
        elif wes_ == 9:
            colorfilm = palettable.wesanderson.GrandBudapest4_5.mpl_colors
            st.header('The Grand Budapest Hotel')
            st.write('"Concierge: And how long will you be staying with us?"')
            st.write('"Mr. Blume: Indefinitely. I’m being sued for divorce."')
            st.image('https://64.media.tumblr.com/fa6793f215402ba9b1cb3d60e2ae7003/tumblr_nno75zVkjk1tvvqeko1_500.jpg')
        elif wes_ == 10:
            colorfilm = palettable.wesanderson.GrandBudapest5_5.mpl_colors
            st.header('The Grand Budapest Hotel')
            st.write('"It’s quite a thing winning the loyalty of a woman like that for nineteen consecutive seasons."')
            st.image('https://64.media.tumblr.com/c7331c7e44e5361c8e91fde9c5a7244c/tumblr_nqcpolelIQ1tvvqeko1_500.jpg')
        elif wes_ == 11:
            colorfilm = palettable.wesanderson.IsleOfDogs1_5.mpl_colors
            st.header('Isle of Dogs')
            st.write('"We’re a pack of scary indestructible alpha dogs."')
            st.image('https://64.media.tumblr.com/c9c2c7613d0025543ac80831fc0fec77/tumblr_p68u437Uok1tvvqeko1_500.jpg')
        elif wes_ == 12:
            colorfilm = palettable.wesanderson.Moonrise1_5.mpl_colors
            st.header('Moonrise Kingdom')
            st.write('"I love you, but you don’t know what you’re talking about."')
            st.image('https://64.media.tumblr.com/ab2f006632656256433200b3b4acc527/tumblr_n2a0afiXub1tvvqeko1_500.jpg')
        elif wes_ == 13:
            colorfilm = palettable.wesanderson.Moonrise4_5.mpl_colors
            st.header('Moonrise Kingdom')
            st.write('"Coming Soon."')
            st.image('https://64.media.tumblr.com/e5bb629e152d5039f00b2f8c7f389f62/tumblr_n2mkj1MMan1tvvqeko1_500.jpg')
        elif wes_ == 14:
            colorfilm = palettable.wesanderson.Moonrise6_5.mpl_colors
            st.header('Moonrise Kingdom')
            st.write('"I\'m not that strong of a swimmer, so I swear a life-preserver."')
            st.write('"I think it’s a good policy to get in the habit, anyway."')
            st.image('https://64.media.tumblr.com/30b2db15b10f5c1f2f1d8a1743a7f360/tumblr_noavg5MNSK1tvvqeko1_500.jpg')
        elif wes_ == 15:
            colorfilm = palettable.wesanderson.Moonrise7_5.mpl_colors
            st.header('You are in Moonrise Kingdom')
            st.write('"Access denied!"')
            st.image('https://64.media.tumblr.com/783a825df1cc7ecf634f88a5b65d5e18/tumblr_o0stq0luWG1tvvqeko1_500.jpg')
        elif wes_ == 16:
            colorfilm = palettable.wesanderson.Royal2_5.mpl_colors
            st.header('The Royal Tenenbaums')
            st.write('"Anybody interested in grabbing a couple of burgers and hittin’ the cemetery?"')
            st.image('https://64.media.tumblr.com/4164ab93fe0460c0b74943b968b6902f/tumblr_nm317bLRIq1tvvqeko1_540.jpg')
        elif wes_ == 17:
            colorfilm = palettable.wesanderson.Royal3_5.mpl_colors
            st.header('The Royal Tenenbaums')
            st.write('"I’ve always been considered an asshole for about as long as I can remember. That’s just my style.""')
            st.image('https://64.media.tumblr.com/128220d4b95d59844a127191ff788c64/tumblr_nvalkkOQbi1tvvqeko1_500.jpg')
        elif wes_ == 18:
            colorfilm = palettable.wesanderson.Zissou_5.mpl_colors
            st.header('The Life Aquatic with Steve Zissou')
            st.write('"Don’t point that gun at him, he’s an unpaid intern."')
            st.image('https://64.media.tumblr.com/ed917df6263b2c492f66df6ae87bd44c/tumblr_n2mkm0i38a1tvvqeko1_500.jpg')
        elif wes_ == 19:
            colorfilm = palettable.wesanderson.Chevalier_4.mpl_colors
            st.header('Hotel Chevalier')

            st.image('https://64.media.tumblr.com/4cd00a745676f9d493de41fbeed40a25/tumblr_n2a0907yB81tvvqeko1_500.jpg')
        elif wes_ == 20:
            colorfilm = palettable.wesanderson.Darjeeling1_4.mpl_colors
            st.header('The Darjeeling Limited')
            st.write('"Jack: I wonder if the three of us would’ve been friends in real life. Not as brothers, but as people."')
            st.image('https://64.media.tumblr.com/b20707749539baa98296d340c983acf1/tumblr_n2q5ltDdbI1tvvqeko1_500.jpg')
        elif wes_ == 21:
            colorfilm = palettable.wesanderson.GrandBudapest1_4.mpl_colors
            st.header('The Grand Budapest Hotel')

            st.image('https://64.media.tumblr.com/71b714a5160b65952a70424dda442c77/tumblr_n2jlhuffzk1tvvqeko1_500.jpg')
        elif wes_ == 22:
            colorfilm = palettable.wesanderson.GrandBudapest2_4.mpl_colors
            st.header('The Grand Budapest Hotel')
            st.write('"M. Gustave: You see, there are still faint glimmers of civilization left in this barbaric slaughterhouse that was once known as humanity."')
            st.image('https://64.media.tumblr.com/ed0b34c7d0ecaffea780728b0e6f4d1d/tumblr_nixlusS2eB1tvvqeko1_500.jpg')
        elif wes_ == 23:
            colorfilm = palettable.wesanderson.IsleOfDogs3_4.mpl_colors
            st.header('Isle of Dogs')
            st.write('"That crook! He’s stealing the re-election again! Let’s go!"')
            st.image('https://64.media.tumblr.com/ddef8260192e8a01eb19fde47d5b5c2d/tumblr_ppuvplYIrk1tvvqeko1_1280.jpg')
        elif wes_ == 24:
            colorfilm = palettable.wesanderson.Mendl_4.mpl_colors
            st.header('The Grand Budapest Hotel, no Mendl\'s')

            st.image('https://64.media.tumblr.com/8afa731262bf3f0d652f41bd7e7e7aa0/tumblr_n2bh1svDwb1tvvqeko1_500.jpg')
        elif wes_ == 25:
            colorfilm = palettable.wesanderson.Moonrise2_4.mpl_colors
            st.header('Moonrise Kingdom')
            st.write('"Sam: Why do you always use binoculars?"')
            st.write('"Suzy: It helps me see things closer. Even if they’re not very far away. I pretend it’s my magic power."')
            st.image('https://64.media.tumblr.com/b7a9d8bd99a533200b8af98720c13fb7/tumblr_n2h3z4H9L21tvvqeko1_500.jpg')
        elif wes_ == 26:
            colorfilm = palettable.wesanderson.Moonrise3_4.mpl_colors
            st.header('Moonrise Kingdom')
            st.write('"Suzy - I’ve always wanted to be an orphan. Most of my favorite characters are."')
            st.image('https://64.media.tumblr.com/e9c308846c8fcb814ec9702d843b314c/tumblr_n2jl0rIlVx1tvvqeko1_500.jpg')
        elif wes_ == 27:
            colorfilm = palettable.wesanderson.Royal1_4.mpl_colors
            st.header('The Royal Tenenbaums')
            st.write('"Royal O Reilly Tenenbaum, 1932 - 2001."')
            st.image('https://64.media.tumblr.com/b7e803f6fbbffa373d3aaf0822511560/tumblr_n2mlvkb0v61tvvqeko1_500.jpg')
        elif wes_ == 28:
            colorfilm = palettable.wesanderson.GrandBudapest3_6.mpl_colors
            st.header('The Grand Budapest Hotel')
            st.write('"M. Gustave: Mendl’s is the best."')
            st.image('https://64.media.tumblr.com/972176fb90a241b7f20f7338b823742d/tumblr_nkhay2zZlo1tvvqeko1_500.jpg')
        elif wes_ == 29:
            colorfilm = palettable.wesanderson.IsleOfDogs2_6.mpl_colors
            st.header('Isle of Dogs')
            st.write('"Be advised that small dogs still pose a threat to the livelihood of Megasaki City. Do not underestimate their size."')
            st.image('https://64.media.tumblr.com/8ecd6d58c020fc06f7949a482ac08c5e/tumblr_p6ntz9dxlH1tvvqeko1_500.jpg')
        elif wes_ == 30:
            colorfilm = palettable.wesanderson.Moonrise5_6.mpl_colors
            st.header('Moonrise Kingdom')
            st.write('"Our daughter’s been abducted by one of these beige lunatics!"')
            st.image('https://64.media.tumblr.com/6dae6ff40086c3738c0e70a8fbf61102/tumblr_nn3xm8Lx3J1tvvqeko1_500.jpg')



        #3D plot of the detected color palette

        st.header('Your color palette in the RGB colorspace:')
        dfx = pd.DataFrame(k_means_cluster_centers_, columns = ('Red','Green','Blue'))
        dfx['hex'] = dfx.apply(lambda x: mpl.colors.to_hex([ x['Red'], x['Green'], x['Blue']]), axis=1)
        fig = px.scatter_3d(dfx, x='Red', y='Green', z='Blue', color_discrete_map='identity', color='hex')
        st.plotly_chart(fig)

        #3D plot of the wes anderson movie color palette
        st.header('The Wes Anderson film color palette:')
        dfw = pd.DataFrame(colorfilm, columns = ('Red','Green','Blue'))
        dfw['hex'] = dfw.apply(lambda x: mpl.colors.to_hex([ x['Red'], x['Green'], x['Blue']]), axis=1)
        figw = px.scatter_3d(dfw, x='Red', y='Green', z='Blue', color_discrete_map='identity', color='hex')
        st.plotly_chart(figw)

        #3D plot of 10% of the entire list of detected colors

        if st.button('Click to show the 3D plot of the full list of detected colors'):
            dft = pd.DataFrame(pixel_flat/255., columns = ('Red','Green','Blue'))
            dft['hex'] = dft.apply(lambda x: mpl.colors.to_hex([ x['Red'], x['Green'], x['Blue']]), axis=1)
            figt = px.scatter_3d(dft.sample(frac=0.1), x='Red', y='Green', z='Blue', color_discrete_map='identity', color='hex')
            st.plotly_chart(figt)



if __name__ == '__main__':
    main()

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
