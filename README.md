# peebot

<img src="peebot.png" height="100" alt="peebot">

peebot is a Discord bot for bringing the horrors of AI chatbots straight to your Discord server.

## Supported backends

### openai_chat

Use the gpt-3.5-turbo model for a ChatGPT-like experience.

#### Model parameters

```toml
# https://platform.openai.com/docs/api-reference/chat/create
temperature = 1.0           # 0.0...2.0
top_p = 1.0                 # 0.0...1.0
presence_penalty = 0.0      # -2.0...2.0
frequency_penalty = 0.0     # -2.0...2.0
```

### spellbook

You're on your own for this one.

#### Model parameters

```toml
# None!
```

## Setup guide

1. Create a forum channel on your Discord server. The bot should be allowed to embed links and post messages in the forum channel.

1. Configure the bot with a TOML file. It should look something like this:

    ```toml
    discord_token = "your-discord-token-here"
    parent_channel_id = 12345 # Your forum channel goes here.

    [backends."gpt-3.5"]
    type = "openai_chat"
    api_key = "your-openai-token-here"
    model = "gpt-3.5-turbo"
    ```

    The first backend listed will be the default backend.

1. Set up tags in your forum channels, if required. For instance:

    - **multi:** Designates the channel as a multi-user chatroom. In multi-user mode, the backend will be prompted with additional contextual information about who said what. Additionally, **all messages will be sent to the backend**, not just ones mentinoing the bot!
    - **use [backend name]:** Allows users to select which backend they want to use. This should match the backends in the config file.

## User guide

To get started, create a forum thread. The title of the forum thread doesn't matter, but the first post should be the system prompt to the bot, for instance telling it how to act.

> **Note:** You can include an optional section after the system prompt, split by `---`, for model parameters.
>
> For instance:
>
> ```
> You are really rude and mean all the time.
> ---
> temperature = 1.4
> ```
>
> The valid parameters depend on which backend you've selected, though. You probably don't need to touch this unless you really know what you're doing.

You can then get the bot to respond by either @mentioning it or replying to one of its message with @ mention on.

### Commands

-   **/forget:** Insert a break in a chat log. Any further responses from the bot will not read past this point. If you want to selectively make the bot ignore messages, react to any messages you'd like it to ignore with the ❌.

-   **/inject:** Just make the bot say something directly.

-   **/injectsystem:** Inject an additional system prompt at the current point in the chat log. You probably don't need to use this.
