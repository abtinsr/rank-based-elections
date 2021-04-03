import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests

#############################################
####### IMPORT AND CLEAN DATA
#############################################

def import_and_clean_data():
    
    # BEST PARTY PREFERENCES
    df_bp = ( # bp = Best Party (Preferences)
        pd.read_csv('../Data/best_party.csv', sep='\t', encoding='latin') # Tab-separated with header from SCB
        .reset_index()   
    )
    
    df_bp.columns = df_bp.iloc[0]
    df_bp = (
        df_bp.drop(0)
        .drop(['kön', 'ålder'], axis=1)
    )
    
    # NEXT BEST PARTY PREFERENCES
    df_nbp = ( # nbp = Next Best Party (Preferences)
    pd.read_csv('../Data/next_best_party.csv', sep='\t', encoding='latin') # Tab-separated with header from SCB
    .reset_index()
    )
    
    df_nbp.columns = df_nbp.iloc[0]
    df_nbp = (
        df_nbp.drop(0)
        .query('partisympati not in ("hela väljarkåren", "ingen sympati/vet ej")')
    )
    
    return(df_bp, df_nbp)

#############################################
####### RETURN DATA FOR A CERTAIN DATE
#############################################

def select_data(df_bp, df_nbp, date):
    
    df_nbp = df_nbp[['partisympati', 'näst bästa parti', f'{date}']]
    
    df_bp = df_bp[['parti', f'{date}']]
    
    df_nbp.columns = ['best_party', 'next_best_party', 'redistribution_share']
    df_bp.columns = ['best_party', 'current_votes']
    
    df = (
        pd.merge(df_bp, df_nbp)
        .replace('..', 0)
        .astype({'current_votes': float, 'redistribution_share': float})
        .assign(redistribution_share=lambda x: x['redistribution_share'] / 100)
        .assign(initial_votes=lambda x: x['current_votes'].astype(float))
        .assign(redistributed_votes=0)
    )
    
    return(df)

#############################################
####### RETURN VOTES IN CURRENT RESULTS
#############################################

def current_results(data):
    top_list = (
        data[['best_party', 'current_votes', 'redistributed_votes']]
        .groupby('best_party')
        .max()
        .reset_index()
        .sort_values('current_votes', ascending=False)
        .reset_index()
        .drop(columns=['index'])
    )
    return(top_list)

#############################################
####### RETURN PARTY WITH CURRENTLY FEWEST VOTES
#############################################

def bottom_party_name(data):
    party = (
        current_results(data)
        .query('current_votes != 0')
        .best_party.iloc[-1] # The party with the fewest votes. 
    )
    return(party)

#############################################
####### RETURN THE CURRENTLY HIGHEST VOTE SHARE
#############################################

def top_party_votes_share(data):
    party = (
        current_results(data)
        .assign(current_votes=lambda x: x['current_votes'] + x['redistributed_votes']) # To count the totals.
        .sort_values('current_votes', ascending=False)
        .current_votes.iloc[0] # The party with the most votes. 
    )
    return(party)

#############################################
####### REDISTRIBUTE VOTES FROM THE BOTTOM PARTY
#############################################

def redistribute_bottom_points(data):
    
    # Identify which party has the least votes - it will be the next to have its votes redistributed. 
    bottom_party = bottom_party_name(data)
    top_vote_share = top_party_votes_share(data)
    
    print("############################################################")
    print(f'Party "{bottom_party}" has been redistributed.') 
    print(f'The highest party vote share is currently {top_vote_share.round(1)}%.') 
    print("############################################################")
    
    # Calculate how many percentage points each party shall get from the bottom party. 
    points_to_redistribute = (
        data[data.best_party == bottom_party]
        .assign(points_to_redistribute=lambda x: x['initial_votes'] * x['redistribution_share'].astype(float))
        .drop(columns=['best_party', 'current_votes', 'redistribution_share', 'initial_votes', 'redistributed_votes'])
        .rename({'next_best_party': 'best_party'}, axis=1) # Rename, because the points should be merged with the party next-in-line. 
    )
        
    # Fill the "redistributed votes" bucket for each party. 
    new_df = (
        data 
        .merge(points_to_redistribute, how='left')
        .assign(redistributed_votes=lambda x: x['redistributed_votes'] + x['points_to_redistribute'].astype(float))
        .drop(columns=['points_to_redistribute'])
    )
    
    # Reset the "current votes" bucket for the bottom party. 
    new_df.loc[new_df.best_party == bottom_party, 'current_votes'] = 0
    
    return(new_df)

#############################################
####### SIMULATE RANK-BASED ELECTION
#############################################
    
def simulateRankBasedElection(data):
    
    # For each party, calculate the share of votes that have no next-best preference. These are the "diehard" voters. 
    diehard_votes = (
    data.copy()
    .query('next_best_party == "inget parti" or next_best_party == "vet ej/uppgift saknas"') # Select the people with no or unknown second-best parties. 
    .assign(diehard_votes=lambda x: x['initial_votes'] * x['redistribution_share'].astype(float)) # Calculate their share of the vote. 
    .drop(columns=['current_votes', 'next_best_party', 'redistribution_share', 'initial_votes'])
    .groupby('best_party')
    .aggregate({'diehard_votes':'sum'}) # Summarise their vote share per party. 
    .reset_index()
    )
    
    # Run through the votes redistribution algorithm. 
    loop_count = 1
    while top_party_votes_share(data) < 50 and loop_count < len(current_results(data)):
        data = redistribute_bottom_points(data)
        loop_count += 1
    
    # Clean the data for the remaining party or parties. 
    data = (
    data.drop(columns=['next_best_party', 'redistribution_share'])
    .groupby('best_party')
    .aggregate({'current_votes': 'mean', 'redistributed_votes': 'mean', 'initial_votes': 'mean'})
    .reset_index()
    )
    
    # Merge with the unbudged votes to get the final election tally. 
    data = pd.merge(data, diehard_votes, how='outer') 
    data.loc[data.current_votes == 0, 'current_votes'] = data.diehard_votes # Those parties that have been culled - and thus have zero current votes - get their unbudged votes back.
    data = data.assign(current_votes=lambda x: x['current_votes'] + x['redistributed_votes']) # Then, they get their redistributed votes added to their current votes tally. 
    
    return(data)

#############################################
####### DEFINE POLITICAL BLOCS
#############################################

def bloc(party):
    return {
        'V': 'Socialdemokratin',
        'SD': 'SD',
        'M': 'Alliansen', 
        'C': 'Alliansen',
        'L': 'Alliansen', 
        'KD': 'Alliansen',
        'S': 'Socialdemokratin', 
        'MP': 'Socialdemokratin',
        'övriga': 'övriga'
    }[party]