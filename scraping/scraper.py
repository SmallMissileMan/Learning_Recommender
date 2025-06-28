import streamlit as st
import pandas as pd
from googleapiclient.discovery import build
import time

# --- YouTube API Config ---
api_key = st.secrets["YOUTUBE_API_KEY"]
youtube = build('youtube', 'v3', developerKey=api_key)

# --- Function to search channel by name and get uploads playlist ID ---
def get_uploads_playlist_id_by_name(channel_name):
    try:
        res = youtube.search().list(
            q=channel_name,
            type="channel",
            part="id",
            maxResults=1
        ).execute()
        if not res['items']:
            return None

        channel_id = res['items'][0]['id']['channelId']
        ch_res = youtube.channels().list(
            part="contentDetails",
            id=channel_id
        ).execute()
        return ch_res['items'][0]['contentDetails']['relatedPlaylists']['uploads'], channel_id
    except Exception as e:
        st.warning(f"⚠️ Could not get playlist for channel {channel_name}\n{e}")
        return None, None

# --- Function to fetch videos from a playlist ---
def fetch_videos_from_playlist(playlist_id, channel_name, max_results=300):
    videos = []
    next_page_token = None
    while len(videos) < max_results:
        res = youtube.playlistItems().list(
            part='snippet',
            playlistId=playlist_id,
            maxResults=min(50, max_results - len(videos)),
            pageToken=next_page_token
        ).execute()

        for item in res['items']:
            title = item['snippet']['title']
            description = item['snippet']['description']
            video_id = item['snippet']['resourceId']['videoId']
            url = f"https://www.youtube.com/watch?v={video_id}"
            published_at = item['snippet']['publishedAt']
            videos.append({
                "Channel": channel_name,
                "Resource Name": title,
                "Description": description,
                "Resource URL": url,
                "Published At": published_at
            })

        next_page_token = res.get('nextPageToken')
        if not next_page_token:
            break
        time.sleep(0.3)
    return videos

# --- List of channels to search ---
channel_names = [
    "Apna College",
    "CodeWithHarry",
    "Kunal Kushwaha",
    "Jenny's Lectures",
    "Neso Academy",
    "Gate Smashers",
    "5 Minutes Engineering",
    "Harkirat Singh",
    "Neeraj Walia"
]

all_data = []

for name in channel_names:
    st.write(f"Fetching from {name}...")
    playlist_id, channel_id = get_uploads_playlist_id_by_name(name)
    if playlist_id:
        try:
            vids = fetch_videos_from_playlist(playlist_id, name, max_results=300)
            all_data.extend(vids)
        except Exception as e:
            st.error(f"❌ Failed to fetch from {name}\n{e}")

# --- Save Dataset ---
df = pd.DataFrame(all_data)
df.to_csv("learning_resources.csv", index=False)
st.success(f"✅ Saved {len(df)} resources to learning_resources.csv")
