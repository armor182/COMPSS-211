# Accessing Data: Some Preliminary Considerations

Whenever you're trying to get information from the web, it's very important to first know whether you're accessing it through appropriate means.

The UC Berkeley library has some excellent resources on this topic. Here is a flowchart that can help guide your course of action:

<center><img src="../../img/scraping_flowchart.png" alt="flowchart" width="700"></center>

You can see the library's licensed sources [here](http://guides.lib.berkeley.edu/text-mining).

# What is an API?

This lesson is about APIs. You may have heard this terminology in a variety of programming settings. What exactly does it mean?

* "API" stands for **Application Programming Interface**.

* Suppose you're at a restaurant. **You (the user)** want something from the **kitchen (a database or service)**, but you can't just walk into the kitchen and grab what you need. If every customer did that, it would be chaotic. **You need a waiter to act as the intermediary**. You tell the waiter what you want using **the menu** - a set of standardized rules, or "orders" - and the waiter brings you back exactly what you ordered. **You can think of the menu and waiter as the API**.

* The most common type of API you'll encounter is a Web API. This lets your program ask for information from websites and services over the internet - like asking Reddit for posts, asking Twitter for tweets, or asking a weather service for today's forecast.

* For example, imagine Reddit has millions of posts stored in their databases. Rather than giving everyone direct access to their servers (which would be chaotic and insecure), Reddit provides an API. This API is like a controlled doorway that lets you politely ask: "Hey Reddit, can I see the top 10 posts from r/programming?" and Reddit sends back just that information.

* Most web APIs follow a style called REST (Representational State Transfer). Don't worry about the fancy name. **The key thing is that REST APIs work using familiar web addresses (URLs), just like the ones you type into your browser**.

* What makes REST APIs so convenient is that you can request data using URLs that look almost like normal website addresses. For example, instead of visiting `reddit.com/r/programming` in your browser, your program might visit `api.reddit.com/r/programming` to get the raw data about that subreddit.

## Web APIs Are All Around You

Consider a simple Google search:

<center><img src="../../img/google_search.png" alt="flowchart" width="800"></center>

Ever wonder what all that extra stuff in the address bar was all about?  In this case, the full address is Google's way of sending a query to its databases asking requesting information related to the search term "golden state warriors". 

<center><img src="../../img/google_link.png" alt="flowchart" width="800"></center>

In fact, it looks like Google makes its query by taking the search terms, separating each of them with a "+", and appending them to the link "https://www.google.com/#q=".  Therefore, we should be able to actually change our Google search by adding some terms to the URL and following the general format:

<center><img src="../../img/google_link_change.png" alt="flowchart" width="800"></center>

Using REST APIs is essentially formatting these URLs so that you can get the response you want.

## The Building Blocks

Let's break down a couple key components:

* **URL (Web Address)**: The location where you're requesting data from. Just like google.com takes you to Google's homepage, an API URL like `api.reddit.com/r/programming` takes you to programming subreddit data. Making a request to an API URL is called making an "API call."
* **GET Requests**: The most common way to ask for information from an API. When you type a URL into your browser or click a link, you're making a GET request - essentially asking "Can I GET this information please?"
* **Other HTTP Methods**: APIs can also handle other types of requests like POST (sending new data), PUT (updating existing data), and DELETE (removing data). But for now, we'll focus on GET requests since that's by far the most common API request you'll use.

When you browse the web normally, you're already using this system! Every webpage you visit involves your browser making GET requests to servers.

## API Examples

- [**Reddit**](https://www.reddit.com/dev/api/):
Used for pulling Reddit data, posting status updates, and more. 

- [**Spotify**](https://developer.spotify.com/):
Access to rich song data data such as valence, energy, and danceability metrics.

-  [**Watson IBM Natural Language Inference API**](https://cloud.ibm.com/apidocs/natural-language-understanding):
Use state of the art NLP models to analyze text sentiment, extract named entities, and classify text.

## API or Web Scraping?

When deciding between using an API or web scraping, you should consider both the method's legality and efficiency. APIs provide structured, authorized access to data, often with clear documentation and rate limits to manage server load. 

Web scraping, on the other hand, involves extracting data from web pages, which may violate a site's terms of service or lead to challenges in navigating complex page structures. 

While scraping can be useful when no API is available, APIs are generally the preferred method for accessing web data due to their reliability and compliance with legal standards. However, sometimes APIs cost money, or have restrictions on the number of queries you can issue.

# The New York Times API

Now, we turn to our main case study of the API portion: the New York Times API.

The New York Times offers a treasure trove of data about their articles that is easily accessible and available for free! We'll now get set up with API keys so that we can make some API calls to the NYT servers.

## Getting API Access

For most APIs, a key or other user credentials are required for any database querying.  Generally, this requires that you register with the organization. Go to the [NYT Developer Page](http://developer.nytimes.com/) and create an account:

<center><img src="../../img/nytimes_start.png" alt="flowchart" width="600"></center>

Most APIs are set up for developers, so you'll likely be asked to register an "application".  All this really entails is coming up with a name for your project, and providing your real name, organization, and email. Not all APIs are this simple to sign up for - in fact, it is increasingly common for companies to charge users for access to APIs.

## Getting your API Keys

Once you've successfully registered, you will be assigned one or more keys, tokens, or other credentials that must be supplied to the server as part of any API call you make.  To make sure that users aren't abusing their data access privileges (e.g. by making many rapid queries), each set of keys will be given several **rate limits** governing the total number of calls that can be made over certain intervals of time.  For the NYT Article API, we have relatively generous rate limits: 10 calls per minute and 4,000 calls per day.

1. Login with your new username and password.

2. Click on your email in the top right corner and you'll see a dropdown menu that says **Apps**

3. Click on **Apps** and then click on the **+ New App** button.

4. You'll see the page where you'll be prompted to add a name for your App. You can call it anything. Then click enable on the APIs that are enabled in the screenshot. You can enable them all but make sure you at least enable the ones on the screenshot. 

<center><img src="../../img/nytimes_app.png" alt="flowchart" width="600"></center>

5. You'll see an API key next to your App ID. Have that key ready to copy into the first notebook.

<center><img src="../../img/nytimes_key.png" alt="flowchart" width="600"></center>

Proceed to the `week06_web-apis/01_api_lesson.md` notebook with your API key to make some calls!
