from flask import render_template
from flask_dir import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
from flask import request
import numpy as np 
import pandas as pd

import os 
path = "/".join(os.path.realpath(__file__).split("/")[:-2])
# print 'path', path
import uuid


import sys
sys.path.append('./vectorsearch/')
import vectorsearch
df_businesses = pd.read_pickle(path+'/input/yelp_academic_dataset_business.pickle')

user = 'carlson' #add your username here (same as previous postgreSQL)            
host = 'localhost'
dbname = 'birth_db'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user)






@app.route('/')
@app.route('/index')
def index():
    return render_template("input.html")
    #return render_template("index.html",
    #   title = 'Home', user = { 'nickname': 'Miguel' },
    #   )

@app.route('/db')
def birth_page():
    sql_query = """                                                             
                SELECT * FROM birth_data_table WHERE delivery_method='Cesarean'\
;                                                                               
                """
    query_results = pd.read_sql_query(sql_query,con)
    births = ""
    print query_results[:10]
    for i in range(0,10):
        births += query_results.iloc[i]['birth_month']
        births += "<br>"
    return births

@app.route('/db_fancy')
def cesareans_page_fancy():
    sql_query = """
               SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean';
                """
    query_results=pd.read_sql_query(sql_query,con)
    births = []
    for i in range(0,query_results.shape[0]):
        births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
    return render_template('cesareans.html',births=births)


@app.route('/input')
def cesareans_input():
    return render_template("input.html")


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

@app.route('/output')
def cesareans_output():
  #pull 'birth_month' from input field and store it
  
  review_text = request.args.get('text_review')


  # SearchBusinesses(review)
  top_n = 5 # Number of topics to choose for top of list. 
  rev_topic = np.array(vectorsearch.GetDocTopic(review_text))
  # Get the top few topics for this review. 
  top_n_topics = rev_topic.argsort()[-top_n:][::-1]
  # print rev_topic # Print the topic vector. 

  topic_listings = [" ".join(vectorsearch.GetTopicWords(topic, ))  for topic in top_n_topics]
  top_bus_id, top_bus_sim = vectorsearch.FindBusinessSimilarityLDA(rev_topic, business_ids=None)
  print topic_listings

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
      vectorsearch.visualize_topic(bus_topic_vec, num_topics=top_n, save_path='/home/carlson/web/'+img_path, top_topics=top_n_topics)
      # Append to list that gets passed to web page...
      top_businesses.append(dict(bus_id=bus_id, similarity=top_bus_sim[i], image_path='http://planck.ucsc.edu/'+img_path,
                            bus_name=df_businesses.name[df_businesses.business_id==bus_id].values[0]))



  


  return render_template("output.html", review_text=review_text, topic_listings=topic_listings, top_businesses=top_businesses,
                          image_path_query='http://planck.ucsc.edu/'+img_path_query)
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