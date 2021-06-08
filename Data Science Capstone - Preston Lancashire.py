#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

#!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab
#from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

from bs4 import BeautifulSoup

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
#import folium # map rendering library

print('Libraries imported.')


# In[2]:


pip install geopy


# In[3]:


from geopy.geocoders import Nominatim


# In[4]:


import pandas as pd

df = pd.read_csv("Preston_Postcode_Sample_Data.csv")
de = df[["Postcode","Neighbourhood","Latitude","Longitude"]]
de.head()


# In[5]:


d1 = de[de['Postcode'].str.contains("PR0 ")]
d2 = de[de['Postcode'].str.contains("PR1 ")]
d3 = de[de['Postcode'].str.contains("PR2 ")]

frames = [d1,d2,d3]

result = pd.concat(frames)
result


# In[6]:


preston_data = result.sample(n=400)
preston_data


# In[7]:


address = 'Preston, UK'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Preston are {}, {}.'.format(latitude, longitude))


# In[8]:


import folium # map rendering library


# In[9]:


# create map of preston using latitude and longitude values

map_preston = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, label in zip(preston_data['Latitude']
                                        ,preston_data['Longitude']
                                                                ,preston_data['Neighbourhood']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_preston)  
    
map_preston
map_preston.save("Map of Preston.jpg")


# In[10]:


CLIENT_ID = 'ACSF133PHSFG5TBKGW12HYLRB1UW5GCBHAFQIULJXSK44FTI' # your Foursquare ID
CLIENT_SECRET = 'EZFCLE3T1CODGFMV5F2YPSPI1E12SI31Y23HZIKFMZV4FUHA' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version
LIMIT = 100 # A default Foursquare API limit value

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[12]:


neighbourhood_latitude = preston_data.loc[300, 'Latitude'] # neighbourhood latitude value
neighbourhood_longitude = preston_data.loc[300
    , 'Longitude'] # neighbourhood longitude value

neighbourhood_name = preston_data.loc[300
    , 'Neighbourhood'] # neighbourhood name

print('Latitude and longitude values of {} are {}, {}.'.format(neighbourhood_name, 
                                                               neighbourhood_latitude, 
                                                               neighbourhood_longitude))


# In[13]:


# type your answer here

LIMIT = 100 # limit of number of venues returned by Foursquare API

radius = 500 # define radius

url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighbourhood_latitude, 
    neighbourhood_longitude, 
    radius, 
    LIMIT)


# In[14]:


results = requests.get(url).json()
results


# In[15]:


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[16]:


venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()


# In[17]:


print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))


# In[18]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighbourhood', 
                  'Neighbourhood Latitude', 
                  'Neighbourhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[19]:


# type your answer here
preston_venues = getNearbyVenues(names=preston_data['Neighbourhood'],
                                   latitudes=preston_data['Latitude'],
                                   longitudes=preston_data['Longitude']
                                  )


# In[20]:


print(preston_venues.shape)
preston_venues.head()


# In[21]:


preston_venues.groupby('Neighbourhood').count()


# In[22]:


print('There are {} uniques categories.'.format(len(preston_venues['Venue Category'].unique())))


# In[23]:


# one hot encoding
preston_onehot = pd.get_dummies(preston_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighbourhood column back to dataframe
preston_onehot['Neighbourhood'] = preston_venues['Neighbourhood'] 

# move neighbourhood column to the first column
fixed_columns = [preston_onehot.columns[-1]] + list(preston_onehot.columns[:-1])
preston_onehot = preston_onehot[fixed_columns]

preston_onehot.head()


# In[24]:


preston_onehot.shape


# In[25]:


preston_grouped = preston_onehot.groupby('Neighbourhood').mean().reset_index()
preston_grouped.head()


# In[26]:


preston_grouped.shape


# In[27]:


num_top_venues = 5

for hood in preston_grouped['Neighbourhood']:
    print("----"+hood+"----")
    temp = preston_grouped[preston_grouped['Neighbourhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[28]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[29]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighbourhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighbourhoods_venues_sorted = pd.DataFrame(columns=columns)
neighbourhoods_venues_sorted['Neighbourhood'] = preston_grouped['Neighbourhood']

for ind in np.arange(preston_grouped.shape[0]):
    neighbourhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(preston_grouped.iloc[ind, :], num_top_venues)

neighbourhoods_venues_sorted.head()


# In[30]:


# set number of clusters
kclusters = 5

preston_grouped_clustering = preston_grouped.drop('Neighbourhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(preston_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# In[31]:


# add clustering labels
neighbourhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

preston_merged = preston_data

# merge preston_grouped with preston_data to add latitude/longitude for each neighbourhood
preston_merged1 = preston_merged.join(neighbourhoods_venues_sorted.set_index('Neighbourhood'), on='Neighbourhood')

#preston_merged1 = pd.merge(left=survey_sub, right=species_sub, left_on='species_id', right_on='species_id')

preston_merged1.head() # check the last columns!


# In[32]:


preston_merged2 = preston_merged1.dropna()
preston_merged2.head()


# In[33]:


import matplotlib.cm as cm
import matplotlib.colors as colors

# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]
#rainbow = ['green','silver','red','blue','black']

#rimes = [1,2,3,4,5]

#add markers to the map
markers_colors = []
for lat, lon, poi, cluster, primes in zip(preston_merged2['Latitude'], preston_merged2['Longitude'], preston_merged2['Neighbourhood'], preston_merged2['Cluster Labels'], preston_merged2['Cluster Labels']):
        label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
        folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[int(cluster)],
        fill=True,
        fill_color=rainbow[int(cluster)],
        fill_opacity=0.7).add_to(map_clusters)
map_clusters


# 

# In[34]:


preston_merged2.loc[preston_merged2['Cluster Labels'] == 0, preston_merged2.columns[[1] + list(range(5, preston_merged2.shape[1]))]]


# In[35]:


preston_merged2.loc[preston_merged2['Cluster Labels'] == 1, preston_merged2.columns[[1] + list(range(5, preston_merged2.shape[1]))]]


# In[36]:


preston_merged2.loc[preston_merged2['Cluster Labels'] == 2, preston_merged2.columns[[1] + list(range(5, preston_merged2.shape[1]))]]


# In[37]:


preston_merged2.loc[preston_merged2['Cluster Labels'] == 3, preston_merged2.columns[[1] + list(range(5, preston_merged2.shape[1]))]]


# In[38]:


preston_merged2.loc[preston_merged2['Cluster Labels'] == 4, preston_merged2.columns[[1] + list(range(5, preston_merged2.shape[1]))]]


# In[ ]:




