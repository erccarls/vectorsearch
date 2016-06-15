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


import sys
sys.path.append('./vectorsearch/')
import vectorsearch
df_businesses = pd.read_pickle(path+'/input/yelp_academic_dataset_business_SF.pickle')



city_state_list = sorted(pickle.load(open(path+'/output/city_state_list.pickle', 'rb')))[1:]



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

  top_bus_id, top_bus_sim = vectorsearch.FindBusinessSimilarityLDA(rev_topic, business_ids=bus_ids_in_city_state, method='cos')
  print "Similarity took", time.time()-start, "seconds" 
  #print topic_listings

  # Visualize the search query.....
  img_path_query = '/images/insight/query_'+str(uuid.uuid4()) + '.png'
  vectorsearch.visualize_topic(rev_topic, num_topics=top_n, save_path='/home/carlson/web/'+img_path_query)
  # Find the top businesses.
  top_businesses = [] 
  for i, bus_id in enumerate(top_bus_id):
      # This is the full topic array for the business. 
      bus_topic_vec = vectorsearch.bus_lda_topics[vectorsearch.bus_lda_topics.business_id==bus_id].topic_vector.values[0]

      img_path = '/images/insight/'+bus_id+'.png'
      print 'Generating image ', img_path
      lat = df_businesses.latitude[df_businesses.business_id==bus_id].values[0]
      lon = df_businesses.longitude[df_businesses.business_id==bus_id].values[0]

      vectorsearch.visualize_topic(bus_topic_vec, num_topics=top_n, save_path='/home/carlson/web/'+img_path, top_topics=top_n_topics)
      # Append to list that gets passed to web page...
      top_businesses.append(dict(bus_id=bus_id, similarity=top_bus_sim[i], image_path='http://planck.ucsc.edu/'+img_path,
                            bus_name=df_businesses.name[df_businesses.business_id==bus_id].values[0],
                            lat=lat, lon=lon))



  centroid_lat = np.average([biz['lat'] for biz in top_businesses])
  centroid_lon = np.average([biz['lon'] for biz in top_businesses])

  # Generate map....
  map_path = img_path[:-4]+'.html'
  print "\nPATH TO MAP, lat, lon", map_path, '\n', centroid_lat, centroid_lon
  map_osm = folium.Map(location=[centroid_lat, centroid_lon], zoom_start=13, detect_retina=True, 
                    tiles='http://tile.stamen.com/watercolor/{z}/{x}/{y}.jpg', attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://creativecommons.org/licenses/by-sa/3.0">CC BY SA</a>.')               
  map_osm.add_tile_layer(tile_url='http://tile.stamen.com/toner-labels/{z}/{x}/{y}.png', attr='labels',
                         active=True, overlay=True)

  for business in top_businesses:

    html = r'''<div align="center"> <font size="4"><b>'''+business['bus_name']+'''</b></font> <br><img src="'''+business['image_path']+'''" alt="NOPE" style="width:250px;height:250px;"></div>'''
    iframe = folium.element.IFrame(html=html,width=300,height=300)
    popup = folium.Popup(html=iframe)
    
    icon = folium.Icon(color="blue", icon="ok")
    marker = folium.Marker(location=[business['lat'], business['lon']], popup=popup, icon=icon)
    map_osm.add_children(marker)

  map_osm.save('/home/carlson/web/'+map_path)




  return render_template("output.html", review_text=review_text, topic_listings=topic_listings, top_businesses=top_businesses,
                          image_path_query='http://planck.ucsc.edu/'+img_path_query, map_path='http://planck.ucsc.edu/'+map_path,
                          city_state_list=city_state_dict)
  #return render_template("output.html", births=[1,5,2,], the_results=" ")



  # try:
  # 	patient = request.args.get('birth_month')
  # except:
  # 	print 'Error'
  # print patient
  #   #just select the Cesareans  from the birth dtabase for the month that the user inputs
  # query = "SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean' AND birth_month='%s'" % patient
  # print query
  # query_results=pd.read_sql_query(query,con)
  # print query_results
  # births = []
  # for i in range(0,query_results.shape[0]):
  #     births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
  #     the_result = ''
  # return render_template("output.html", births = births, the_result = the_result)