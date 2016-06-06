import sys
sys.path.append('../vectorsearch/')
import nltk_helper


def parse_query(query):
    '''
    Parse the search string which must follow a dict specification format, 
    *Must* use semicolons to separate entries.
    e.g. query = "dive bar:5 ; music : [1,4] ; expensive : -1 "
    
    query : str
        Search string
    '''
    try: 
        terms = query.split(';')
        query_dict = {}
        for term in terms: 
            key, val = term.split(':')
            key = nltk_helper.clean_nltk(key)[0]

            if "[" in val and "]" in val: 
                val = val.replace('[','').replace(']','').split(",")
                try:
                    val = [float(weight) for weight in val]
                    query_dict[key] = val
                except: 
                    print 'Warning, value for key %s could not be parsed'%key
            # If just a floating point number..
            else: 
                try: 
                    val = float(val)
                    query_dict[key] = val
                except: 
                    print 'Warning, value for key %s could not be parsed'%key
    except: 
        print 'Sorry... invalid query try again.  example: "dive bar:5 ; music : [1,4] ; expensive : -1 "'
    return query_dict



if __name__=='__main__':
    print parse_query('dive bar : [1,10] ; music:1 ')
    
    