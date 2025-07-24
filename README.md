# ✈️ Trafella - Your AI Travel Companion

Trafella is a web-based travel planning application built with Streamlit that acts as your personal AI travel assistant. It helps you plan your perfect trip by providing real-time flight options, detailed destination information, personalized itineraries, and recommendations for hotels and restaurants.

## demo
https://trafella.streamlit.app/

## ✨ Features

- **Personalized Travel Planning:** Get customized travel plans based on your destination, travel dates, trip theme, and personal preferences.
- **Real-Time Flight Search:** Fetches the top 3 cheapest flight options using the SerpAPI Google Flights integration.
- **AI-Powered Recommendations:** Utilizes Google's Gemini model to provide insightful information about your destination, including attractions, cultural norms, and safety tips.
- **Detailed Itineraries:** Generates a day-by-day itinerary with suggested activities, hotels, and restaurants.
- **Multi-Language Support:** In-built translation from English to Bahasa Indonesia.
- **Secure API Key Handling:** Ensures your API keys are stored securely in the browser session and are not exposed.

## ⚙️ Architecture & Structure

Trafella is a single-script Streamlit application with a clear and modular structure.

- **`trafella.py`**: The main and only script, which contains all the application logic.
  - **UI:** The user interface is built entirely with Streamlit.
  - **Flight Search:** It uses the `google-search-results` library to query the SerpAPI Google Flights endpoint.
  - **AI Agents:** It leverages the `agno` library to create three specialized AI agents (`Researcher`, `Planner`, `Hotel & Restaurant Finder`), all powered by the Gemini model.
  - **Translation:** It uses the `transformers` library with a `Helsinki-NLP` model for local translation.

- **`requirements.txt`**: Lists all the necessary Python dependencies.
- **`LICENSE`**: The project is licensed under the MIT License.

## 🚀 Getting Started

### Prerequisites

- Python 3.12 or higher
- Pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/naufal1910/trafella_prototype.git
    cd trafella_prototype
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

Trafella requires API keys for SerpAPI and Google AI. The application will prompt you to enter these keys in the sidebar when you first run it.

1.  **SerpAPI Key:** Get your key from [SerpAPI](https://serpapi.com/manage-api-key).
2.  **Google API Key:** Get your key from [Google AI Studio](https://aistudio.google.com/app/u/1/apikey).

*Your keys are only stored in your browser's session and are never saved on the server.*

## Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run trafella.py
    ```

2.  **Enter your API keys** in the sidebar as prompted.

3.  **Fill in your travel details:**
    - Departure and destination cities (using IATA codes, e.g., `CGK` for Jakarta).
    - Travel duration, theme, and preferred activities.
    - Departure and return dates.

4.  **Click "Generate Travel Plan"** and let the AI agents do the work!

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
