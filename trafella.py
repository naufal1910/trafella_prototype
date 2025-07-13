"""
Trafella - Your AI Travel Companion
----------------------------------
Trafella is an intelligent travel planning assistant that helps you discover and plan your perfect trip by providing:
- ✈️ Flight options and booking information
- 🌍 Comprehensive destination insights
- 📅 Personalized daily itineraries
- 🏨 Curated hotel and restaurant recommendations

Powered by:
1. Streamlit for intuitive web interface
2. Google's Gemini AI for intelligent planning
3. SerpAPI for real-time flight data
4. Advanced travel planning algorithms
"""

import streamlit as st
import json
import os
import logging
from datetime import datetime
from serpapi.google_search import GoogleSearch 
from agno.agent import Agent
from agno.tools.serpapi import SerpApiTools
from agno.models.google import Gemini

# Configure logging
log_filename = datetime.now().strftime("%Y%m%d") + "-trafella.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Trafella")
logger.info("Application started")

# Set up Streamlit UI with a travel-friendly theme
logger.info("Setting up Streamlit UI")
st.set_page_config(page_title="🌍 AI Travel Planner", layout="wide")
st.markdown(
    """
    <style>
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #ff5733;
        }
        .subtitle {
            text-align: center;
            font-size: 20px;
            color: #555;
        }
        .stSlider > div {
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and subtitle
logger.info("Setting up Streamlit UI")
st.markdown('<h1 class="title">✈️ AI-Powered Travel Planner</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Plan your dream trip with AI! Get personalized recommendations for flights, hotels, and activities.</p>', unsafe_allow_html=True)

# User Inputs Section
logger.info("Collecting user inputs")
st.markdown("### 🌍 Where are you headed?")
source = st.text_input("🛫 Departure City (IATA Code):", "CGK")  # Example: CGK for Jakarta
destination = st.text_input("🛬 Destination (IATA Code):", "KUL")  # Example: KUL for Kuala Lumpur

st.markdown("### 📅 Plan Your Adventure")
num_days = st.slider("🕒 Trip Duration (days):", 1, 14, 5)
travel_theme = st.selectbox(
    "🎭 Select Your Travel Theme:",
    ["💑 Couple Getaway", "👨‍👩‍👧‍👦 Family Vacation", "🏔️ Adventure Trip", "🧳 Solo Exploration"]
)

# Divider for aesthetics
st.markdown("---")

st.markdown(
    f"""
    <div style="
        text-align: center; 
        padding: 15px; 
        background-color: #ffecd1; 
        border-radius: 10px; 
        margin-top: 20px;
    ">
        <h3>🌟 Your {travel_theme} to {destination} is about to begin! 🌟</h3>
        <p>Let's find the best flights, stays, and experiences for your unforgettable journey.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Function to format ISO datetime string to a human-readable format
def format_datetime(iso_string):
    """
    Converts an ISO datetime string to a human-readable format.
    
    Args:
    iso_string (str): The ISO datetime string to be formatted.
    
    Returns:
    str: The formatted datetime string.
    """
    try:
        dt = datetime.strptime(iso_string, "%Y-%m-%d %H:%M")
        return dt.strftime("%b-%d, %Y | %I:%M %p")  # Example: Mar-06, 2025 | 6:20 PM
    except:
        return "N/A"

activity_preferences = st.text_area(
    "🌍 What activities do you enjoy? (e.g., relaxing on the beach, exploring historical sites, nightlife, adventure)",
    "Relaxing on the beach, exploring historical sites"
)

departure_date = st.date_input("Departure Date")
return_date = st.date_input("Return Date")

# Sidebar Setup
st.sidebar.title("🌎 Travel Assistant")
st.sidebar.subheader("Personalize Your Trip")

# Travel Preferences
budget = st.sidebar.radio("💰 Budget Preference:", ["Economy", "Standard", "Luxury"])
flight_class = st.sidebar.radio("✈️ Flight Class:", ["Economy", "Business", "First Class"])
hotel_rating = st.sidebar.selectbox("🏨 Preferred Hotel Rating:", ["Any", "3⭐", "4⭐", "5⭐"])

SERPAPI_KEY = "867c0d73d26dae01b24aec9e2114baac336672f1eefbb0c50510ef505e14488d"
GOOGLE_API_KEY = "AIzaSyBjzdwpXUc_h1h4mk6Zw22J7TFxmfJQAyc"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

params = {
        "engine": "google_flights",
        "departure_id": source,
        "arrival_id": destination,
        "outbound_date": str(departure_date),
        "return_date": str(return_date),
        "currency": "IDR",
        "hl": "en",
        "api_key": SERPAPI_KEY
    }

# Function to fetch flight data using SerpAPI
def fetch_flights(source, destination, departure_date, return_date):
    """
    Fetches flight data using SerpAPI.
    
    Args:
    source (str): The departure city IATA code.
    destination (str): The arrival city IATA code.
    departure_date (str): The departure date.
    return_date (str): The return date.
    
    Returns:
    dict: The flight data.
    """
    logger.info(f"Fetching flights: {source} to {destination} on {departure_date} to {return_date}")
    
    # Create a safe version of params for logging (without API key)
    log_params = {
        "engine": "google_flights",
        "departure_id": source,
        "arrival_id": destination,
        "outbound_date": str(departure_date),
        "return_date": str(return_date),
        "currency": "IDR",
        "hl": "en"
    }
    logger.info(f"Sending SerpAPI request with parameters: {log_params}")
    
    params = {
        "engine": "google_flights",
        "departure_id": source,
        "arrival_id": destination,
        "outbound_date": str(departure_date),
        "return_date": str(return_date),
        "currency": "IDR",
        "hl": "en",
        "api_key": SERPAPI_KEY
    }
    
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Log API response status
        status = results.get('search_metadata', {}).get('status', 'unknown')
        logger.info(f"SerpAPI response status: {status}")
        
        # Log number of flights found
        num_flights = len(results.get('best_flights', []))
        logger.info(f"Received {num_flights} flights from SerpAPI")
        
        # Log truncated response for debugging
        logger.debug(f"SerpAPI response (truncated): {str(results)[:200]}...")
        
        return results
    except Exception as e:
        logger.error(f"SerpAPI call failed: {str(e)}")
        return {}

# Function to extract top 3 cheapest flights from flight data
def extract_cheapest_flights(flight_data):
    """
    Extracts the top 3 cheapest flights from the flight data.
    
    Args:
    flight_data (dict): The flight data.
    
    Returns:
    list: The top 3 cheapest flights.
    """
    logger.info("Extracting cheapest flights")
    best_flights = flight_data.get("best_flights", [])
    sorted_flights = sorted(best_flights, key=lambda x: x.get("price", float("inf")))[:3]  # Get top 3 cheapest
    logger.info(f"Top 3 cheapest flights extracted: {len(sorted_flights)} flights found")
    return sorted_flights

# Initialize AI agents for travel planning
# Researcher agent gathers destination information
researcher = Agent(
    name="Researcher",
    instructions=[
        "Identify the travel destination specified by the user.",
        "Gather detailed information on the destination, including climate, culture, and safety tips.",
        "Find popular attractions, landmarks, and must-visit places.",
        "Search for activities that match the user’s interests and travel style.",
        "Prioritize information from reliable sources and official travel guides.",
        "Provide well-structured summaries with key insights and recommendations."
    ],
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[SerpApiTools(api_key=SERPAPI_KEY)],
    add_datetime_to_instructions=True,
)

# Planner agent creates detailed itineraries
planner = Agent(
    name="Planner",
    instructions=[
        "Gather details about the user's travel preferences and budget.",
        "Create a detailed itinerary with scheduled activities and estimated costs.",
        "Ensure the itinerary includes transportation options and travel time estimates.",
        "Optimize the schedule for convenience and enjoyment.",
        "Present the itinerary in a structured format."
    ],
    model=Gemini(id="gemini-2.0-flash-exp"),
    add_datetime_to_instructions=True,
)

# Hotel/Restaurant finder agent locates accommodations and dining
hotel_restaurant_finder = Agent(
    name="Hotel & Restaurant Finder",
    instructions=[
        "Identify key locations in the user's travel itinerary.",
        "Search for highly rated hotels near those locations.",
        "Search for top-rated restaurants based on cuisine preferences and proximity.",
        "Prioritize results based on user preferences, ratings, and availability.",
        "Provide direct booking links or reservation options where possible."
    ],
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[SerpApiTools(api_key=SERPAPI_KEY)],
    add_datetime_to_instructions=True,
)

# Main execution block: Generate travel plan when button is clicked
if st.button("🚀 Generate Travel Plan"):
    """
    Generates a travel plan based on user input.
    """

    logger.info("Generate Travel Plan button clicked")
    with st.spinner("✈️ Fetching best flight options..."):
        logger.info("Starting flight search")
        flight_data = fetch_flights(source, destination, departure_date, return_date)
        cheapest_flights = extract_cheapest_flights(flight_data)

    # AI Processing
    with st.spinner("🔍 Researching destination..."):
        logger.info("Starting destination research")
        research_prompt = (
            f"Research the best attractions and activities in {destination} for a {num_days}-day {travel_theme.lower()} trip. "
            f"The traveler enjoys: {activity_preferences}. Budget: {budget}. Flight Class: {flight_class}. "
            f"Hotel Rating: {hotel_rating}."
        )
        research_results = researcher.run(research_prompt, stream=False)
        logger.info(f"Research results: {research_results.content[:100]}...")

    with st.spinner("🏨 Finding hotels & restaurants..."):
        logger.info("Starting hotel/restaurant search")
        hotel_restaurant_prompt = (
            f"Find the best hotels and restaurants near popular attractions in {destination} for a {travel_theme.lower()} trip. "
            f"Budget: {budget}. Hotel Rating: {hotel_rating}. Preferred activities: {activity_preferences}."
        )
        hotel_restaurant_results = hotel_restaurant_finder.run(hotel_restaurant_prompt, stream=False)
        logger.info(f"Hotel/restaurant results: {hotel_restaurant_results.content[:100]}...")

    with st.spinner("🗺️ Creating your personalized itinerary..."):
        logger.info("Generating itinerary")
        planning_prompt = (
            f"Based on the following data, create a {num_days}-day itinerary for a {travel_theme.lower()} trip to {destination}. "
            f"The traveler enjoys: {activity_preferences}. Budget: {budget}. Flight Class: {flight_class}. Hotel Rating: {hotel_rating}. "
            f"Research: {research_results.content}. Flights: {json.dumps(cheapest_flights)}. Hotels & Restaurants: {hotel_restaurant_results.content}."
        )
        itinerary = planner.run(planning_prompt, stream=False)
        logger.info(f"Itinerary generated: {itinerary.content[:100]}...")

    # Display Results
    st.subheader("✈️ Cheapest Flight Options")
    if cheapest_flights:
        cols = st.columns(len(cheapest_flights))
        for idx, flight in enumerate(cheapest_flights):
            with cols[idx]:
                airline_logo = flight.get("airline_logo", "")
                airline_name = flight.get("airline", "Unknown Airline")
                price = flight.get("price", "Not Available")
                total_duration = flight.get("total_duration", "N/A")
                
                flights_info = flight.get("flights", [{}])
                departure = flights_info[0].get("departure_airport", {})
                arrival = flights_info[-1].get("arrival_airport", {})
                airline_name = flights_info[0].get("airline", "Unknown Airline") 
                
                departure_time = format_datetime(departure.get("time", "N/A"))
                arrival_time = format_datetime(arrival.get("time", "N/A"))
                
                departure_token = flight.get("departure_token", "")

                if departure_token:
                    params_with_token = {
                        **params,
                        "departure_token": departure_token  # Add the token here
                    }
                    search_with_token = GoogleSearch(params_with_token)
                    results_with_booking = search_with_token.get_dict()

                    booking_options = results_with_booking['best_flights'][idx]['booking_token']

                booking_link = f"https://www.google.com/travel/flights?tfs="+booking_options if booking_options else "#"
                print(booking_link)
                # Flight card layout
                st.markdown(
                    f"""
                    <div style="
                        border: 2px solid #ddd; 
                        border-radius: 10px; 
                        padding: 15px; 
                        text-align: center;
                        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                        background-color: #f9f9f9;
                        margin-bottom: 20px;
                    ">
                        <img src="{airline_logo}" width="100" alt="Flight Logo" />
                        <h3 style="margin: 10px 0;">{airline_name}</h3>
                        <p><strong>Departure:</strong> {departure_time}</p>
                        <p><strong>Arrival:</strong> {arrival_time}</p>
                        <p><strong>Duration:</strong> {total_duration} min</p>
                        <h2 style="color: #008000;">💰 {price}</h2>
                        <a href="{booking_link}" target="_blank" style="
                            display: inline-block;
                            padding: 10px 20px;
                            font-size: 16px;
                            font-weight: bold;
                            color: #fff;
                            background-color: #007bff;
                            text-decoration: none;
                            border-radius: 5px;
                            margin-top: 10px;
                        ">🔗 Book Now</a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.warning("⚠️ No flight data available.")

    st.subheader("🏨 Hotels & Restaurants")
    st.write(hotel_restaurant_results.content)

    st.subheader("🗺️ Your Personalized Itinerary")
    st.write(itinerary.content)

    logger.info("Travel plan generated successfully")
    st.success("✅ Travel plan generated successfully!")

    # Log final results
    logger.info(f"Cheapest flights: {json.dumps(cheapest_flights, indent=2)}")
    logger.info(f"Hotels & Restaurants: {hotel_restaurant_results.content}")
    logger.info(f"Itinerary: {itinerary.content}")
