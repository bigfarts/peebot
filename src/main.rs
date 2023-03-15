mod openai;
mod unichunk;

use clap::Parser;
use futures_util::StreamExt;
use handlebars::Renderable;

const MAX_MESSAGE_BUFFER: usize = 2048;

#[derive(Debug)]
struct ChatSettings {
    system_message_format: handlebars::Template,
    user_message_format: handlebars::Template,
    parameters: ChatParameters,
}

impl std::str::FromStr for ChatSettings {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts = s
            .split("\n---\n")
            .into_iter()
            .map(|p| Some(p))
            .chain(std::iter::repeat(None))
            .take(3)
            .collect::<Vec<_>>();

        Ok(ChatSettings {
            system_message_format: handlebars::Template::compile(parts[0].unwrap())?,
            user_message_format: handlebars::Template::compile(parts[1].unwrap_or("{{content}}"))?,
            parameters: parts[2].map_or_else(
                || Ok(ChatParameters::default()),
                |v| toml::from_str::<ChatParameters>(v),
            )?,
        })
    }
}

#[derive(serde::Deserialize, Default, Debug)]
struct ChatParameters {
    temperature: Option<f64>,
    top_p: Option<f64>,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    single_user: bool,
}

#[derive(Debug)]
struct Thread {
    primary_message: serenity::model::channel::Message,
    messages: std::collections::BTreeMap<
        serenity::model::id::MessageId,
        serenity::model::channel::Message,
    >,
}

impl Thread {
    async fn new(
        http: impl AsRef<serenity::http::Http>,
        id: serenity::model::id::ChannelId,
    ) -> Result<Self, serenity::Error> {
        let primary_message = id.message(&http, id.0).await?;
        let mut messages = std::collections::BTreeMap::new();

        let mut messages_it = Box::pin(id.messages_iter(&http)).take(MAX_MESSAGE_BUFFER);
        while let Some(message) = messages_it.next().await {
            let message = message?;
            if message.id.0 == id.0 {
                continue;
            }
            messages.insert(message.id, message);
        }

        Ok(Self {
            primary_message,
            messages,
        })
    }
}

struct Handler {
    handlebars: handlebars::Handlebars<'static>,
    nicknames: tokio::sync::Mutex<
        std::collections::HashMap<
            (serenity::model::id::GuildId, serenity::model::id::UserId),
            String,
        >,
    >,
    me_id: parking_lot::Mutex<serenity::model::id::UserId>,
    tokenizer: tiktoken_rs::CoreBPE,
    config: Config,
    chat_client: openai::ChatClient,
    threads: tokio::sync::Mutex<
        std::collections::HashMap<
            serenity::model::id::ChannelId,
            std::sync::Arc<tokio::sync::Mutex<Option<Thread>>>,
        >,
    >,
}

static RESOLVE_MESSAGE_REGEX: once_cell::sync::Lazy<regex::Regex> =
    once_cell::sync::Lazy::new(|| {
        regex::Regex::new(
            r"<@!?(?P<user_id>\d+)>|<a?:(?P<emoji_name>\w+):\d+>|<#(?P<channel_id>\d+)>",
        )
        .unwrap()
    });

static STRIP_SINGLE_USER_REGEX: once_cell::sync::Lazy<regex::Regex> =
    once_cell::sync::Lazy::new(|| regex::Regex::new(r"^\s*<@!?(?P<user_id>\d+)>\s*").unwrap());

impl Handler {
    async fn resolve_display_name(
        &self,
        http: impl AsRef<serenity::http::Http>,
        guild_id: serenity::model::id::GuildId,
        user_id: serenity::model::id::UserId,
    ) -> Result<String, serenity::Error> {
        let mut nicknames = self.nicknames.lock().await;
        match nicknames.entry((guild_id, user_id)) {
            std::collections::hash_map::Entry::Occupied(e) => Ok(e.get().clone()),
            std::collections::hash_map::Entry::Vacant(e) => {
                let member = http.as_ref().get_member(guild_id.0, user_id.0).await?;
                Ok(e.insert(member.display_name().into_owned()).clone())
            }
        }
    }

    async fn resolve_message(
        &self,
        http: impl AsRef<serenity::http::Http>,
        guild_id: serenity::model::id::GuildId,
        content: &str,
    ) -> Result<String, serenity::Error> {
        let mut s = String::new();
        let mut last_index = 0;

        for capture in RESOLVE_MESSAGE_REGEX.captures_iter(content) {
            let m = capture.get(0).unwrap();

            s.push_str(&content[last_index..m.start()]);

            let repl = if let Some(subm) = capture.name("user_id") {
                let user_id = subm.as_str().parse::<u64>().unwrap();
                self.resolve_display_name(&http, guild_id, user_id.into())
                    .await?
            } else if let Some(subm) = capture.name("emoji_name") {
                format!(":{}:", subm.as_str())
            } else if let Some(subm) = capture.name("channel_id") {
                let _channel_id = subm.as_str().parse::<u64>().unwrap();
                "#".to_string()
            } else {
                "".to_string()
            };
            s.push_str(&repl);
            last_index = m.end();
        }
        s.push_str(&content[last_index..]);
        Ok(s)
    }
}

#[async_trait::async_trait]
impl serenity::client::EventHandler for Handler {
    async fn ready(
        &self,
        _ctx: serenity::client::Context,
        data_about_bot: serenity::model::gateway::Ready,
    ) {
        *self.me_id.lock() = data_about_bot.user.id;
    }

    async fn guild_create(
        &self,
        _ctx: serenity::client::Context,
        guild: serenity::model::guild::Guild,
    ) {
        if let Err(e) = (|| async {
            let mut threads = self.threads.lock().await;
            for thread in guild.threads.iter() {
                if thread.parent_id
                    != Some(serenity::model::id::ChannelId(
                        self.config.parent_channel_id,
                    ))
                {
                    continue;
                }

                log::info!("thread {} scheduled for load", thread.id);
                threads.insert(
                    thread.id,
                    std::sync::Arc::new(tokio::sync::Mutex::new(None)),
                );
            }

            Ok::<_, anyhow::Error>(())
        })()
        .await
        {
            log::error!("error in guild_create: {:?}", e);
        }
    }

    async fn thread_create(
        &self,
        ctx: serenity::client::Context,
        thread: serenity::model::channel::GuildChannel,
    ) {
        if let Err(e) = (|| async {
            if thread.parent_id
                != Some(serenity::model::id::ChannelId(
                    self.config.parent_channel_id,
                ))
            {
                return Ok(());
            }

            if thread.last_message_id != None {
                return Ok(());
            }

            thread.id.join_thread(&ctx.http).await?;

            let thread_info = Thread::new(&ctx.http, thread.id).await?;
            log::info!(
                "thread {} joined: {:?}",
                thread.id,
                thread_info.primary_message
            );

            let mut threads = self.threads.lock().await;
            threads.insert(
                thread.id,
                std::sync::Arc::new(tokio::sync::Mutex::new(Some(thread_info))),
            );

            Ok::<_, anyhow::Error>(())
        })()
        .await
        {
            log::error!("error in thread_create: {:?}", e);
        }
    }

    async fn thread_update(
        &self,
        _ctx: serenity::client::Context,
        thread: serenity::model::channel::GuildChannel,
    ) {
        if let Err(e) = (|| async {
            if thread.parent_id
                != Some(serenity::model::id::ChannelId(
                    self.config.parent_channel_id,
                ))
            {
                return Ok(());
            }

            let mut threads = self.threads.lock().await;
            if thread.thread_metadata.unwrap().archived {
                log::info!("thread {} archived", thread.id);
                threads.remove(&thread.id);
            } else {
                match threads.entry(thread.id) {
                    std::collections::hash_map::Entry::Occupied(_) => {}
                    std::collections::hash_map::Entry::Vacant(e) => {
                        log::info!("thread {} scheduled for load", thread.id);
                        e.insert(std::sync::Arc::new(tokio::sync::Mutex::new(None)));
                    }
                }
            }

            Ok::<_, anyhow::Error>(())
        })()
        .await
        {
            log::error!("error in thread_update: {:?}", e);
        }
    }

    async fn thread_delete(
        &self,
        _ctx: serenity::client::Context,
        thread: serenity::model::channel::PartialGuildChannel,
    ) {
        if let Err(e) = (|| async {
            let mut threads = self.threads.lock().await;
            log::info!("thread {} deleted", thread.id);
            threads.remove(&thread.id);
            Ok::<_, anyhow::Error>(())
        })()
        .await
        {
            log::error!("error in thread_delete: {:?}", e);
        }
    }

    async fn guild_member_update(
        &self,
        _ctx: serenity::client::Context,
        event: serenity::model::event::GuildMemberUpdateEvent,
    ) {
        if let Err(e) = (|| async {
            let mut nicknames = self.nicknames.lock().await;
            nicknames.insert(
                (event.guild_id, event.user.id),
                event.nick.unwrap_or(event.user.name),
            );
            Ok::<_, anyhow::Error>(())
        })()
        .await
        {
            log::error!("error in guild_member_update: {:?}", e);
        }
    }

    async fn message(
        &self,
        ctx: serenity::client::Context,
        new_message: serenity::model::channel::Message,
    ) {
        if let Err(e) = (|| async {
            let me_id = self.me_id.lock().clone();

            let thread = {
                let threads = self.threads.lock().await;
                let thread = if let Some(thread) = threads.get(&new_message.channel_id) {
                    thread
                } else {
                    return Ok(());
                };
                thread.clone()
            };

            let can_reply = thread.try_lock().is_ok();
            if !can_reply {
                new_message
                    .channel_id
                    .send_message(&ctx.http, |m| {
                        m.embed(|e| {
                            e.color(serenity::utils::colours::css::WARNING);
                            e.description("I'm already replying, please wait for me to finish!");
                            e
                        });
                        m
                    })
                    .await?;
            }

            let mut thread = thread.lock().await;

            let thread = if let Some(thread) = thread.as_mut() {
                thread
            } else {
                log::info!("thread {} loaded", new_message.channel_id);
                *thread = Some(Thread::new(&ctx.http, new_message.channel_id).await?);
                thread.as_mut().unwrap()
            };

            thread.messages.insert(new_message.id, new_message.clone());

            if !can_reply || !new_message.mentions_user_id(me_id) {
                return Ok(());
            }

            if let Err(e) = (|| async {
                let settings: ChatSettings = thread.primary_message.content.parse()?;

                let system_message = {
                    let mut output = handlebars::StringOutput::new();

                    #[derive(serde::Serialize)]
                    struct SystemMessageFormatArgs {
                        display_name: String,
                        timestamp: String,
                    }

                    settings.system_message_format.render(
                        &self.handlebars,
                        &handlebars::Context::wraps(SystemMessageFormatArgs {
                            display_name: self
                                .resolve_display_name(
                                    &ctx.http,
                                    new_message.guild_id.unwrap(),
                                    me_id,
                                )
                                .await?,
                            timestamp: new_message
                                .timestamp
                                .with_timezone(&chrono::Utc)
                                .to_rfc3339(),
                        })?,
                        &mut handlebars::RenderContext::new(None),
                        &mut output,
                    )?;

                    openai::Message {
                        role: openai::Role::System,
                        name: None,
                        content: output.into_string()?,
                    }
                };

                let mut messages = vec![];

                for (_, message) in thread.messages.iter() {
                    let mut next_messages = messages.clone();
                    if message.content.is_empty() {
                        continue;
                    }

                    let oai_message = if message.author.id == me_id {
                        openai::Message {
                            role: openai::Role::Assistant,
                            name: None,
                            content: message.content.clone(),
                        }
                    } else {
                        let mut content = message.content.clone();
                        if settings.parameters.single_user {
                            if !message.mentions_user_id(me_id) {
                                continue;
                            }

                            // Need to do some regex replacement.
                            content = STRIP_SINGLE_USER_REGEX
                                .replace(&content, |c: &regex::Captures| {
                                    if serenity::model::id::UserId(
                                        c["user_id"].parse::<u64>().unwrap(),
                                    ) == me_id
                                    {
                                        "".to_string()
                                    } else {
                                        c[0].to_string()
                                    }
                                })
                                .to_string();
                        }
                        content = self
                            .resolve_message(&ctx.http, new_message.guild_id.unwrap(), &content)
                            .await?;

                        let mut output = handlebars::StringOutput::new();

                        #[derive(serde::Serialize)]
                        struct UserMessageFormatArgs {
                            display_name: String,
                            timestamp: String,
                            content: String,
                        }

                        settings.user_message_format.render(
                            &self.handlebars,
                            &handlebars::Context::wraps(UserMessageFormatArgs {
                                display_name: self
                                    .resolve_display_name(
                                        &ctx.http,
                                        new_message.guild_id.unwrap(),
                                        me_id,
                                    )
                                    .await?,
                                timestamp: new_message
                                    .timestamp
                                    .with_timezone(&chrono::Utc)
                                    .to_rfc3339(),
                                content,
                            })?,
                            &mut handlebars::RenderContext::new(None),
                            &mut output,
                        )?;

                        openai::Message {
                            role: openai::Role::User,
                            name: None,
                            content: output.into_string()?,
                        }
                    };

                    next_messages.push(oai_message.clone());
                    next_messages.push(system_message.clone());

                    if openai::count_tokens(&self.tokenizer, &next_messages)
                        > self.config.max_input_tokens as usize
                    {
                        break;
                    }

                    messages.push(oai_message);
                }

                messages.insert(0, system_message);

                let input_tokens = openai::count_tokens(&self.tokenizer, &messages) as u32;

                let req = openai::ChatRequest {
                    messages,
                    stream: true,
                    model: "gpt-3.5-turbo".to_owned(),
                    temperature: settings.parameters.temperature,
                    top_p: settings.parameters.top_p,
                    frequency_penalty: settings.parameters.frequency_penalty,
                    presence_penalty: settings.parameters.presence_penalty,
                    max_tokens: Some(self.config.max_tokens - input_tokens),
                };
                log::info!("{:#?}", req);

                let _typing = new_message.channel_id.start_typing(&ctx.http)?;

                let mut stream = Box::pin(
                    tokio::time::timeout(
                        std::time::Duration::from_secs(30),
                        self.chat_client.request(&req),
                    )
                    .await
                    .map_err(|e| anyhow::format_err!("timed out: {}", e))??,
                );

                let mut chunker = unichunk::Chunker::new(2000);
                while let Some(chunk) =
                    tokio::time::timeout(std::time::Duration::from_secs(30), stream.next())
                        .await
                        .map_err(|e| anyhow::format_err!("timed out: {}", e))?
                {
                    let chunk = chunk?;
                    let delta = &chunk.choices[0].delta;
                    let content = if let Some(content) = delta.content.as_ref() {
                        content
                    } else {
                        continue;
                    };
                    for c in chunker.push(&content) {
                        new_message
                            .channel_id
                            .send_message(&ctx.http, |m| {
                                m.content(&c);
                                m
                            })
                            .await?;
                    }
                }

                let c = chunker.flush();
                if !c.is_empty() {
                    new_message
                        .channel_id
                        .send_message(&ctx.http, |m| {
                            m.content(&c);
                            m
                        })
                        .await?;
                }

                Ok::<_, anyhow::Error>(())
            })()
            .await
            {
                new_message
                    .channel_id
                    .send_message(&ctx.http, |m| {
                        m.embed(|e| {
                            e.title("Error");
                            e.color(serenity::utils::colours::css::DANGER);
                            e.description(format!("{:?}", e));
                            e
                        });
                        m
                    })
                    .await?;
                return Err(e);
            }

            Ok::<_, anyhow::Error>(())
        })()
        .await
        {
            log::error!("error in message: {:?}", e);
        }
    }

    async fn message_update(
        &self,
        _ctx: serenity::client::Context,
        new_event: serenity::model::event::MessageUpdateEvent,
    ) {
        if let Err(e) = (|| async {
            let thread = {
                let threads = self.threads.lock().await;
                let thread = if let Some(thread) = threads.get(&new_event.channel_id) {
                    thread
                } else {
                    return Ok(());
                };
                thread.clone()
            };

            let mut thread = thread.lock().await;
            let thread = if let Some(thread) = thread.as_mut() {
                thread
            } else {
                return Ok(());
            };

            let message = if new_event.id.0 == new_event.channel_id.0 {
                &mut thread.primary_message
            } else if let Some(message) = thread.messages.get_mut(&new_event.id) {
                message
            } else {
                return Ok(());
            };

            if let Some(x) = new_event.attachments {
                message.attachments = x
            }
            if let Some(x) = new_event.content {
                message.content = x
            }
            if let Some(x) = new_event.edited_timestamp {
                message.edited_timestamp = Some(x)
            }
            if let Some(x) = new_event.mentions {
                message.mentions = x
            }
            if let Some(x) = new_event.mention_everyone {
                message.mention_everyone = x
            }
            if let Some(x) = new_event.mention_roles {
                message.mention_roles = x
            }
            // if let Some(x) = new_event.mention_channels {
            //     message.mention_channels = x
            // }
            if let Some(x) = new_event.pinned {
                message.pinned = x
            }
            if let Some(x) = new_event.flags {
                message.flags = Some(x)
            }
            if let Some(x) = new_event.tts {
                message.tts = x
            }
            if let Some(x) = new_event.embeds {
                message.embeds = x
            }
            // if let Some(x) = new_event.reactions {
            //     message.reactions = x
            // }
            // if let Some(x) = new_event.components {
            //     message.components = x
            // }
            // if let Some(x) = new_event.sticker_items {
            //     message.sticker_items = x
            // }

            Ok::<_, anyhow::Error>(())
        })()
        .await
        {
            log::error!("error in message_update: {:?}", e);
        }
    }
    async fn message_delete(
        &self,
        _ctx: serenity::client::Context,
        channel_id: serenity::model::id::ChannelId,
        deleted_message_id: serenity::model::id::MessageId,
        _guild_id: Option<serenity::model::id::GuildId>,
    ) {
        if let Err(e) = (|| async {
            let thread = {
                let threads = self.threads.lock().await;
                let thread = if let Some(thread) = threads.get(&channel_id) {
                    thread
                } else {
                    return Ok(());
                };
                thread.clone()
            };

            let mut thread = thread.lock().await;
            let thread = if let Some(thread) = thread.as_mut() {
                thread
            } else {
                return Ok(());
            };

            thread.messages.remove(&deleted_message_id);

            Ok::<_, anyhow::Error>(())
        })()
        .await
        {
            log::error!("error in message_delete: {:?}", e);
        }
    }

    async fn message_delete_bulk(
        &self,
        _ctx: serenity::client::Context,
        channel_id: serenity::model::id::ChannelId,
        multiple_deleted_messages_id: Vec<serenity::model::id::MessageId>,
        _guild_id: Option<serenity::model::id::GuildId>,
    ) {
        if let Err(e) = (|| async {
            let thread = {
                let threads = self.threads.lock().await;
                let thread = if let Some(thread) = threads.get(&channel_id) {
                    thread
                } else {
                    return Ok(());
                };
                thread.clone()
            };

            let mut thread = thread.lock().await;
            let thread = if let Some(thread) = thread.as_mut() {
                thread
            } else {
                return Ok(());
            };

            for message_id in multiple_deleted_messages_id {
                thread.messages.remove(&message_id);
            }

            Ok::<_, anyhow::Error>(())
        })()
        .await
        {
            log::error!("error in message_delete_bulk: {:?}", e);
        }
    }
}

#[derive(clap::Parser)]
struct Opts {
    #[clap(default_value = "config.toml")]
    config: std::path::PathBuf,
}

const fn max_input_tokens_default() -> u32 {
    2048
}

const fn max_tokens_default() -> u32 {
    4096
}

#[derive(serde::Deserialize)]
struct Config {
    openai_token: String,
    discord_token: String,
    parent_channel_id: u64,
    #[serde(default = "max_input_tokens_default")]
    max_input_tokens: u32,
    #[serde(default = "max_tokens_default")]
    max_tokens: u32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::builder()
        .filter_module("peebot", log::LevelFilter::Info)
        .init();

    log::info!("hello!");

    let opts = Opts::parse();

    let config = toml::from_str::<Config>(std::str::from_utf8(&std::fs::read(opts.config)?)?)?;

    let chat_client = openai::ChatClient::new(&config.openai_token);

    let intents = serenity::model::gateway::GatewayIntents::default()
        | serenity::model::gateway::GatewayIntents::MESSAGE_CONTENT
        | serenity::model::gateway::GatewayIntents::GUILD_MESSAGES
        | serenity::model::gateway::GatewayIntents::GUILDS
        | serenity::model::gateway::GatewayIntents::GUILD_MEMBERS;

    let mut handlebars = handlebars::Handlebars::default();
    handlebars.register_escape_fn(|s| s.to_owned());

    serenity::client::ClientBuilder::new(&config.discord_token, intents)
        .event_handler(Handler {
            handlebars,
            nicknames: tokio::sync::Mutex::new(std::collections::HashMap::new()),
            me_id: parking_lot::Mutex::new(serenity::model::id::UserId::default()),
            tokenizer: tiktoken_rs::cl100k_base().unwrap(),
            config,
            chat_client,
            threads: tokio::sync::Mutex::new(std::collections::HashMap::new()),
        })
        .await?
        .start()
        .await?;

    Ok(())
}
