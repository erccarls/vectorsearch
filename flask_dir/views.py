from flask import render_template
from flask_dir import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
from flask import request
import numpy as np 
import pandas as pd
import folium
import pickle
import time
import os 
path = "/".join(os.path.realpath(__file__).split("/")[:-2])
# print 'path', path
import uuid
from folium import plugins
from flask import jsonify
import sys
sys.path.append('./vectorsearch/')
import vectorsearch

from bs4 import BeautifulSoup
import re
import urllib2
from wordcloud import WordCloud

city_state_list = sorted(pickle.load(open(path+'/output/city_state_list.pickle', 'rb')))[1:]
df_businesses = pd.read_pickle(path+'/input/yelp_academic_dataset_business_SF.pickle')
bus_reviews = pickle.load(open(path+'/output/reviews_merged_bus.pickle', 'rb'))

from matplotlib import pyplot as plt
from datetime import timedelta
from flask import make_response, request, current_app
from functools import update_wrapper

from multiprocessing import pool 



def crossdomain(origin=None, methods=None, headers=None,
                max_age=21600, attach_to_all=True,
                automatic_options=True):
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, basestring):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, basestring):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        def wrapped_function(*args, **kwargs):
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers
            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            h['Access-Control-Allow-Credentials'] = 'true'
            h['Access-Control-Allow-Headers'] = \
                "Origin, X-Requested-With, Content-Type, Accept, Authorization"
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator





@app.route('/')
@app.route('/index')
def index():
    city_state_dict = [dict(city=city) for city in city_state_list]
    return render_template("input.html", city_state_list=city_state_dict)
    #return render_template("index.html",
    #   title = 'Home', user = { 'nickname': 'Miguel' },
    #   )

@app.route('/db')
def birth_page():
  return render_template("stylish_test.html")


@app.route('/input')
def cesareans_input():
    city_state_list = [dict(city=city) for city in city_state_list]
    return render_template("input.html", city_state_list=city_state_list)


# @app.route('/output')
# def cesareans_output():
#   #pull 'birth_month' from input field and store it
#   #patient = request.args.get('birth_month')
#     #just select the Cesareans  from the birth dtabase for the month that the user inputs
#   #query = """SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean';"""
#   #print query
#   # query_results=pd.read_sql_query(query,con)
#   # print query_results
#   # births = []
#   # for i in range(0,query_results.shape[0]):
#   #     #births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
#   #     the_result = ''
#   # return render_template("output.html", births = births, the_result = the_result)
#   return render_template("output.html")

def get_bus_ids_city_state(city, state):
    return list(df_businesses.business_id[(df_businesses.city.str.lower()==city.lower()) 
                                     & (df_businesses.state.str.lower()==state.lower())].values)


@app.route('/output', )
def cesareans_output():
  #pull 'birth_month' from input field and store it
  
  review_text = request.args.get('text_review')
  city_state_dict = [dict(city=city) for city in city_state_list]
  city, state = request.args.get('sel_city').split(",")
  # SearchBusinesses(review)
  top_n = 5 # Number of topics to choose for top of list. 
  rev_topic = np.array(vectorsearch.GetDocTopic(review_text))
  # Get the top few topics for this review. 
  top_n_topics = rev_topic.argsort()[-top_n:][::-1]
  # print rev_topic # Print the topic vector. 


  bus_ids_in_city_state = get_bus_ids_city_state(city.strip(), state.strip())

  topic_listings = [" ".join(vectorsearch.GetTopicWords(topic, ))  for topic in top_n_topics]

  start = time.time()

  #top_bus_id, top_bus_sim = vectorsearch.FindBusinessSimilarityLDA(rev_topic, business_ids=bus_ids_in_city_state, method='Hel', top_n=30)
  top_bus_id, top_bus_sim = vectorsearch.FindBusinessSimilaritydoc2vec(review_text, bus_ids_in_city_state, top_n=50)
  print "Similarity took", time.time()-start, "seconds" 
  #print topic_listings

  # Check that the names are not already included. 
  names, valid_biz, biz_sims = [], [], [] 
  for i_bus, bus_id in enumerate(top_bus_id):
    name = df_businesses.name[df_businesses.business_id==bus_id].values[0]
    if name not in names:
      names.append(name)
      valid_biz.append(bus_id)
      biz_sims.append(top_bus_sim[i_bus])

  top_bus_id  = valid_biz
  top_bus_sim = biz_sims


  # Visualize the search query.....
  img_path_query = '/images/insight/query_'+str(uuid.uuid4()) + '.png'
  #vectorsearch.visualize_topic(rev_topic, num_topics=top_n, save_path='/home/carlson/web/'+img_path_query)
  # Find the top businesses.
  top_businesses = [] 
  words_paths = [] 

  for i, bus_id in enumerate(top_bus_id[:20]):
      # This is the full topic array for the business. 
      bus_topic_vec = vectorsearch.bus_lda_topics[vectorsearch.bus_lda_topics.business_id==bus_id].topic_vector.values[0]

      img_path = '/images/insight/'+bus_id+'.png'
      #print 'Generating image ', img_path
      lat = df_businesses.latitude[df_businesses.business_id==bus_id].values[0]
      lon = df_businesses.longitude[df_businesses.business_id==bus_id].values[0]
      URL = df_businesses.URL[df_businesses.business_id==bus_id].values[0]
      image_URL = df_businesses.image_URL[df_businesses.business_id==bus_id].values[0]
      words = bus_reviews[bus_id]
      words_paths.append((words, img_path))
      #vectorsearch.visualize_topic(bus_topic_vec, num_topics=top_n, save_path='/home/carlson/web/'+img_path, top_topics=top_n_topics)
      # Append to list that gets passed to web page...
      top_businesses.append(dict(bus_id=bus_id, similarity=top_bus_sim[i], image_path='http://planck.ucsc.edu/'+img_path,
                            bus_name="%i. "%(i+1) + df_businesses.name[df_businesses.business_id==bus_id].values[0],
                            lat=lat, lon=lon, URL=URL, image_URL=image_URL))

  # Generate word clouds
  p = pool.Pool(12)
  p.map(gen_word_cloud, words_paths)
  p.close()
  p.join()


  centroid_lat = np.average([biz['lat'] for biz in top_businesses])
  centroid_lon = np.average([biz['lon'] for biz in top_businesses])

  # Generate map....
  map_path = img_path[:-4]+'.html'
  print "\nPATH TO MAP, lat, lon", map_path, '\n', centroid_lat, centroid_lon
  # map_osm = folium.Map(location=[centroid_lat, centroid_lon], zoom_start=13, detect_retina=True, 
  #                   tiles='stamentoner', attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://creativecommons.org/licenses/by-sa/3.0">CC BY SA</a>.')               

  map_osm = folium.Map(location=[centroid_lat, centroid_lon], zoom_start=13, detect_retina=True, 
                    tiles='http://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png', attr='&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="http://cartodb.com/attributions">CartoDB</a>')               

  # map_osm.add_tile_layer(tile_url='http://tile.stamen.com/toner-labels/{z}/{x}/{y}.png', attr='labels',
  #                        active=True, overlay=True)

  for business in top_businesses[:]:

    html = r'''<div align="center"> <font size="4"><a href="'''+business['URL'] +'''"> <b>'''+business['bus_name']+'''</b></a></font> <br><img src="'''+business['image_path']+'''" alt="NOPE" style="width:250px;height:125px;"></div>'''
    iframe = folium.element.IFrame(html=html,width=300,height=175)
    popup = folium.Popup(html=iframe)
    
    icon = folium.Icon(color="blue", icon="ok")
    marker = folium.Marker(location=[business['lat'], business['lon']], popup=popup, icon=icon)
    map_osm.add_children(marker)


  heatmap_events = [(df_businesses.latitude[df_businesses.business_id==bus_id].values[0], 
                     df_businesses.longitude[df_businesses.business_id==bus_id].values[0], 
                     -top_bus_sim[i]+top_bus_sim[0]) for i, bus_id in enumerate(top_bus_id)]

  lats = sims_array = np.array(heatmap_events)[:,0]
  lons = sims_array = np.array(heatmap_events)[:,1]
  sims_array = np.array(heatmap_events)[:,2]
  scale = top_bus_sim[6]-top_bus_sim[0]
  sims_array = ((1-1/(np.exp(sims_array/scale)+1))*50).astype(np.int32)

  heatmap = [] 
  for i, sim in enumerate(sims_array):
      for j in range(sim):
          heatmap += [[lats[i]+.00001*j, lons[i]]]


  map_osm.add_children(plugins.HeatMap(heatmap, max_zoom=18, radius=25, max_val=20))

  map_osm.save('/home/carlson/web/'+map_path)

  append_mousemove_js('/home/carlson/web/'+map_path)


  return render_template("output.html", review_text=review_text, topic_listings=topic_listings, top_businesses=top_businesses,
                          image_path_query='http://planck.ucsc.edu/'+img_path_query, map_path='http://planck.ucsc.edu/'+map_path,
                          city_state_list=city_state_dict)
  #return render_template("output.html", births=[1,5,2,], the_results=" ")







def distance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km

    dlat = np.deg2rad(lat2-lat1)
    dlon = np.deg2rad(lon2-lon1)
    a = np.sin(dlat/2.) * np.sin(dlat/2.) + np.cos(np.deg2rad(lat1)) \
        * np.cos(np.deg2rad(lat2)) * np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = radius * c

    return d

@app.route('/ajax')
@crossdomain(origin='*')
def query_lat_lon(): 

    #print "Queried... ", request.args

    # Find distance to each business.

    lat1 = float(request.args.get('lat'))
    lon1 = float(request.args.get('lon'))


    lat2, lon2 = df_businesses.latitude.values, df_businesses.longitude.values

    distances = distance((lat1, lon1), (lat2,lon2))

    nearby_bids = df_businesses.business_id[distances<.75].values
    # Threshold
    distances[distances<.075] = .075
    weights = 1./distances[distances<.75]**1.5

    bus_topics = np.array([vectorsearch.bus_lda_topics[vectorsearch.bus_lda_topics.business_id==biz].topic_vector.values[0] 
                                          for biz in nearby_bids])
    try:
      bus_topics = weights*bus_topics.T
      if len(bus_topics.shape)<2:
        return jsonify(lat="")
      print bus_topics.shape
      bus_topics = np.sum(bus_topics, axis=1)
      bus_topics /= np.sum(bus_topics)

      topic_idx = bus_topics.argsort()[-6:][::-1]

      topic_words = ["  ".join(vectorsearch.GetTopicWords(i, n_top_words=3)).title() for i in topic_idx]

      return jsonify(lat=lat1, lon=lon1, mean_topic=list(bus_topics[topic_idx]), topic_words=topic_words)
    except:
      return jsonify(lat="")

def append_mousemove_js(mapfile, ):
    soup = BeautifulSoup(open(mapfile, 'rb'), "lxml")
    
    mapname=""
    for line in open(mapfile, 'rb').readlines():
      if "var map_" in line:
          map_name = line.split()[1]
          break
    
    js='''
      <script>
      var last_mousemove = new Date();
                  '''+map_name+'''.on('mousemove', function(e) {
                      var req = new XMLHttpRequest()
                      req.onreadystatechange = function()
                      {
                          if (req.readyState == 4)
                          {
                              if (req.status != 200)
                              {
                                  //error handling code here
                              }
                              else
                              {

                                  var response = JSON.parse(req.responseText)

                                  // console.log(response.mean_topic);
                                  // console.log(response.topic_words);

                                  var arr = [];

                                  for(var x in response.topic_words){
                                    arr.push( {letter:response.topic_words[x], frequency:response.mean_topic[x]} );
                                  }
                                  
                                  //draw(arr);
                                  parent.postMessage(arr,"*");  //  `*` on any domain         

                              }
                          }
                      }
                      

                      // Rate Limiting
                      //console.log((new Date() - last_mousemove));
                      if ((new Date() - last_mousemove) < 300)
                      {
                          return ; 
                      }
                      last_mousemove = new Date();

                      var lat = e.latlng.lat
                      var lon = e.latlng.lng
                      req.open('GET', 'http://planck.ucsc.edu:5000/ajax?lat='+String(lat)+'&lon='+String(lon))
                      req.setRequestHeader("Content-type", "application/x-www-form-urlencoded")

                      req.send()
                      
                      return false
                  });
      </script>
      </html>
      '''
    
    with open(mapfile, 'wb') as f:
        f.write(str(soup).replace("</html>",js))

import wordcloud as wc 
def gen_word_cloud(words_path):
    text, path = words_path
    
    # Generate a word cloud image
    wordcloud = WordCloud().generate(text)


    # take relative word frequencies into account, lower max_font_size
    wordcloud = WordCloud(width=250, height=125, prefer_horizontal=1, max_font_size=60, max_words=20, 
                          min_font_size=14, relative_scaling=1, background_color=None, mode="RGBA",
                          color_func=wc.get_single_color_func("#242426"), stopwords=['bar', "the", "place"]).generate(text)
    plt.figure(figsize=(2.5,1.25), dpi=100)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig('/home/carlson/web/'+path, transparent=True, bbox_inches='tight')
