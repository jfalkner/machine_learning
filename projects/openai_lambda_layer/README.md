# Use OpenAI/ChatGPT's API with AWS Lambda

ChatGPT is an amazing tool that combines several foundational models for generating text responses, images from text, images to text and many other types of tools. The UI is remarkably easy to use, and with a prompt you can have it write almost anything.

[AWS Lambda](https://aws.amazon.com/pm/lambda/) is an appealing way to host custom APIs in a [very inexpensive fashion](https://aws.amazon.com/lambda/pricing/) (e.g. $0.20 per 1M requests with $0.0000166667 for every GB-second of runtime). You write the code and it'll expose the function to your other AWS services. You can also use AWS Gateway to expose the function as an HTTP API that a webpage can use. AWS Lambda is a great option to pay very little for a web API, includign one that a static website can use. 

## Brief Demo

I setup an AI chat bot that uses my resume on [jaysonfalkner.com](http://jaysonfalkner.com) as a demo of what you can do with some static pages and an AWS Lambda function (this write-up) that uses ChatGPT's API. It has a plain text prompt that asks ChatGPT an question via OpenAI's API then displays it on the page. Instead of a static LinkedIn style resume, you can ask exactly what you want and ChatGPT will summarize a response -- even phrasing the output in any practical or silly way desired.

<img src="jaysonfalkner_website.gif">

Above is based on having a text version of [my resume](https://www.linkedin.com/in/jayson-falkner/) as one chat prompt ([RAG](https://aws.amazon.com/what-is/retrieval-augmented-generation/)-style injection of data), then a followup prompt asking the question about the data plus specifying how the output should be phrased (normal, like a pirate, etc.).

Most of the data is a copy of my LinkedIn profile. If you ask the prompt "Where has Jayson worked previously?". It more or less remakes my LinkedIn profile. You can even prompt ChatGPT to have specific output, such as a bulleted list with one line per job with the compnay name, location, time worked there and job title. 

Other extra snippets support common questions. For example, below is one example.

> If asked about example work, note the projects from Jayson's work history and also note that he has several interesting GitHub repositories with soruce-code at https://github.com/jfalkner. Specific GitHub repositories of interest include an archive of algorithms, mostly from code interviews, at https://github.com/jfalkner/algorithms, and also Jayson's recent machine learning and AI work at https://github.com/jfalkner/machine_learning. Don't forget to mention that this interactive resume is also a decent example using generative AI creatively since the text comes from an LLM and images are made via latent diffusion.

The above snippet is why questions similar to "Where can I find some examples of Jayson's work?" end up having some good, specific suggestions.

> You can find examples of Jayson Falkner's work on his GitHub repositories. Here are a couple of repositories that might be of interest:
>
> 1. Algorithms: This repository contains an archive of algorithms, mostly from code interviews.
>
> 2. Machine Learning: In this repository, you can find Jayson's recent work in machine learning and AI.
> ...

And if clicking the pirate icon (or others), the prompt gets changed to something similar to "Where can I find some examples of Jayson's work? Answer like a pirate.", which remarkably does what you'd expect.

> Arrr matey! If ye be lookin' to plunder through Jayson Falkner's treasure trove of code, set sail to his GitHub repositories, savvy?
>
> 1. Algorithms: Here be a chest full o' algorithms, mostly from code interviews.
>
> 2. Machine Learning: In this repository, ye'll discover Jayson's recent exploits in machine learnin' and AI.
>...

The background images on the site change too via OpenAI's DALL-E image generator. That too uses ChatGPT's API with the same AWS Lambda layer setup in this write-up; however, I'll cover it more in a different article.

## Using OpenAI's API from AWS Lambda

There are two main tasks that you'll need to do.

1. Create an AWS Lambda "Layer" to make OpenAI's ChatGPT API be available
2. Write a Python-based AWS Lambda function that uses the API

Above assumes that you already have an AWS console account and OpenAI account setup. Those topics aren't covered here because it is regular registration for these services. If you need to, just go sign up.

### Create an AWS Lambda "Layer" to make OpenAI's ChatGPT API be available

AWS has documentation for making Layers for Python 


## ChatGPT can actually almost write this all itself!
Here is an example prompt that almost replicates this write-up. The main gaps are that screenshots of AWS's tools aren't included.

> Summarize an example of making an S3 layer for OpenAI's API using the latest version of Python. Give a command-line example of using pip to install OpenAI's API. Give command line steps to create a zip for AWS lambda. Give configuration examples for making an AWS Lambda layer from the ZIP. Given and example AWS Lambda function written in Python that uses the layer and asks ChatGPT prompts that are specified by a user.

It is also fair to note that you'd need to have a decent idea of how to build AWS Lambda functions before you'd know 

