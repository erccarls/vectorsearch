

def get_review_ids(business_id, review):
    '''
    Return all review_ids for a corresponding business. 
    
    Parameters
    ---------------
    
    business_id : list
        business_id of interest. 
        
    review: pandas dataframe containing list of review entries.
    
    Returns
    -------------
    review_ids: list
        review_ids for each review of the corresponding business 
    '''
    return review.review_id[review.business_id==business_id]