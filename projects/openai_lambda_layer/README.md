# Use OpenAI/ChatGPT's API with AWS Lambda

ChatGPT is an amazing tool that combines several foundational models for generating text responses, images from text, images to text and many other types of tools. The UI is remarkably easy to use, and with a prompt you can have it write almost anything.

[AWS Lambda](https://aws.amazon.com/pm/lambda/) is an appealing way to host custom APIs in a [very inexpensive fashion](https://aws.amazon.com/lambda/pricing/) (e.g. $0.20 per 1M requests with $0.0000166667 for every GB-second of runtime). You write the code and it'll expose the function to your other AWS services. You can also use AWS Gateway to expose the function as an HTTP API that a webpage can use. AWS Lambda is a great option to pay very little for a web API, includign one that a static website can use. 

## Brief Demo

I setup an AI chat bot that uses my resume on [jaysonfalkner.com](http://jaysonfalkner.com) as a demo of what you can do with an AWS Lambda function that uses ChatGPT's API. It has a plain text prompt that asks ChatGPT an question via OpenAI's API then displays it on the page. Instead of a static LinkedIn style resume, you can ask exactly what you want and ChatGPT will summarize a response -- even phrasing the output in any practical or silly way desired.

<img src="jaysonfalkner_website.com">


## ChatGPT can actually almost write this all itself!
Here is an example prompt that almost replicates this write-up. The main gaps are that screenshots of AWS's tools aren't included.

> Summarize an example of making an S3 layer for OpenAI's API using the latest version of Python. Give a command-line example of using pip to install OpenAI's API. Give command line steps to create a zip for AWS lambda. Give configuration examples for making an AWS Lambda layer from the ZIP. Given and example AWS Lambda function written in Python that uses the layer and asks ChatGPT prompts that are specified by a user.


