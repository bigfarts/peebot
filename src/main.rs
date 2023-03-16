mod openai;
mod unichunk;

use clap::Parser;
use futures_util::StreamExt;

const MAX_MESSAGE_BUFFER: usize = 2048;

#[derive(Debug, PartialEq)]
enum ThreadMode {
    Single,
    Multi,
}

impl ThreadMode {
    fn from_channel_name(name: &str) -> Self {
        if name.to_lowercase().contains("[multi]") {
            ThreadMode::Multi
        } else {
            ThreadMode::Single
        }
    }
}

#[derive(Debug)]
struct ChatSettings {
    system_message: String,
    model_settings: ModelSettings,
}

static STRIP_TRAILING_WHITESPACE_REGEX: once_cell::sync::Lazy<regex::Regex> = once_cell::sync::Lazy::new(|| regex::Regex::new(r"[ \t]+\n").unwrap());

impl ChatSettings {
    fn new(s: &str) -> Result<Self, anyhow::Error> {
        let s = STRIP_TRAILING_WHITESPACE_REGEX.replace_all(s, "\n");
        let parts = s
            .split("\n---\n")
            .into_iter()
            .map(|p| Some(p))
            .chain(std::iter::repeat(None))
            .take(2)
            .collect::<Vec<_>>();

        Ok(ChatSettings {
            system_message: parts[0].unwrap().to_string(),
            model_settings: parts[1].map_or_else(|| Ok(ModelSettings::default()), |v| toml::from_str::<ModelSettings>(v))?,
        })
    }
}

#[derive(serde::Deserialize, Default, Debug)]
#[serde(default)]
struct ModelSettings {
    temperature: Option<f64>,
    top_p: Option<f64>,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
}

#[derive(Debug)]
struct Thread {
    primary_message: serenity::model::channel::Message,
    messages: std::collections::BTreeMap<serenity::model::id::MessageId, serenity::model::channel::Message>,
    mode: ThreadMode,
}

impl Thread {
    async fn new(
        http: impl AsRef<serenity::http::Http>,
        me_id: serenity::model::id::UserId,
        id: serenity::model::id::ChannelId,
    ) -> Result<Self, serenity::Error> {
        let primary_message = id.message(&http, id.0).await?;
        let mut messages = std::collections::BTreeMap::new();

        let mut messages_it = Box::pin(id.messages_iter(&http)).take(MAX_MESSAGE_BUFFER);
        while let Some(message) = messages_it.next().await {
            let message = message?;
            if message.id.0 == id.0 {
                break;
            }

            if message.author.id == me_id {
                if let Some(interaction) = message.interaction.as_ref() {
                    if interaction.kind == serenity::model::application::interaction::InteractionType::ApplicationCommand
                        && interaction.name == FORGET_COMMAND_NAME
                    {
                        break;
                    }
                }
            }

            if message.kind != serenity::model::channel::MessageType::Regular && message.kind != serenity::model::channel::MessageType::InlineReply {
                continue;
            }

            messages.insert(message.id, message);
        }

        let channel = if let serenity::model::prelude::Channel::Guild(guild_channel) = http.as_ref().get_channel(id.0).await? {
            guild_channel
        } else {
            unreachable!();
        };

        Ok(Self {
            primary_message,
            messages,
            mode: ThreadMode::from_channel_name(&channel.name),
        })
    }
}

struct Resolver {
    display_names: std::collections::HashMap<(serenity::model::id::GuildId, serenity::model::id::UserId), String>,
}

impl Resolver {
    fn new() -> Self {
        Self {
            display_names: std::collections::HashMap::new(),
        }
    }

    fn add_display_name(&mut self, guild_id: serenity::model::id::GuildId, user_id: serenity::model::id::UserId, name: String) {
        self.display_names.insert((guild_id, user_id), name);
    }

    async fn resolve_display_name(
        &mut self,
        http: impl AsRef<serenity::http::Http>,
        guild_id: serenity::model::id::GuildId,
        user_id: serenity::model::id::UserId,
    ) -> Result<String, serenity::Error> {
        match self.display_names.entry((guild_id, user_id)) {
            std::collections::hash_map::Entry::Occupied(e) => Ok(e.get().clone()),
            std::collections::hash_map::Entry::Vacant(e) => {
                let member = http.as_ref().get_member(guild_id.0, user_id.0).await?;
                Ok(e.insert(member.display_name().into_owned()).clone())
            }
        }
    }

    async fn resolve_message(
        &mut self,
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
                self.resolve_display_name(&http, guild_id, user_id.into()).await?
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

struct Handler {
    resolver: tokio::sync::Mutex<Resolver>,
    me_id: parking_lot::Mutex<serenity::model::id::UserId>,
    tokenizer: tiktoken_rs::CoreBPE,
    config: Config,
    chat_client: openai::ChatClient,
    threads: tokio::sync::Mutex<std::collections::HashMap<serenity::model::id::ChannelId, std::sync::Arc<tokio::sync::Mutex<Option<Thread>>>>>,
}

static RESOLVE_MESSAGE_REGEX: once_cell::sync::Lazy<regex::Regex> =
    once_cell::sync::Lazy::new(|| regex::Regex::new(r"<@!?(?P<user_id>\d+)>|<a?:(?P<emoji_name>\w+):\d+>|<#(?P<channel_id>\d+)>").unwrap());

static STRIP_SINGLE_USER_REGEX: once_cell::sync::Lazy<regex::Regex> =
    once_cell::sync::Lazy::new(|| regex::Regex::new(r"^\s*<@!?(?P<user_id>\d+)>\s*").unwrap());

const FORGET_COMMAND_NAME: &str = "forget";

#[async_trait::async_trait]
impl serenity::client::EventHandler for Handler {
    async fn ready(&self, ctx: serenity::client::Context, data_about_bot: serenity::model::gateway::Ready) {
        if let Err(e) = (|| async {
            *self.me_id.lock() = data_about_bot.user.id;
            serenity::model::application::command::Command::create_global_application_command(&ctx.http, |command| {
                command
                    .name(FORGET_COMMAND_NAME)
                    .description("Add a break in the chat log to forget everything before it.")
            })
            .await?;

            Ok::<_, anyhow::Error>(())
        })()
        .await
        {
            log::error!("error in ready: {:?}", e);
        }
    }

    async fn interaction_create(&self, ctx: serenity::client::Context, interaction: serenity::model::application::interaction::Interaction) {
        if let Err(e) = (|| async {
            let app_command = if let Some(app_command) = interaction.application_command() {
                app_command
            } else {
                return Ok(());
            };

            let thread = {
                let threads = self.threads.lock().await;
                let thread = if let Some(thread) = threads.get(&app_command.channel_id) {
                    thread
                } else {
                    return Ok(());
                };
                thread.clone()
            };

            let mut thread = thread.lock().await;

            if app_command.kind == serenity::model::application::interaction::InteractionType::ApplicationCommand
                && app_command.data.name == FORGET_COMMAND_NAME
            {
                log::info!("thread {} flushed for forget", app_command.channel_id);
                app_command
                    .create_interaction_response(&ctx.http, |r| {
                        r.interaction_response_data(|d| {
                            d.embed(|e| {
                                e.color(serenity::utils::colours::css::POSITIVE);
                                e.description("Okay, forgetting everything from here. If you want me to remember, just delete this message.");
                                e
                            });
                            d
                        });
                        r
                    })
                    .await?;
                *thread = None;
            }

            Ok::<_, anyhow::Error>(())
        })()
        .await
        {
            log::error!("error in interaction_create: {:?}", e);
        }
    }

    async fn guild_create(&self, ctx: serenity::client::Context, guild: serenity::model::guild::Guild) {
        if let Err(e) = (|| async {
            let mut threads = self.threads.lock().await;
            for thread in guild.threads.iter() {
                if thread.parent_id != Some(serenity::model::id::ChannelId(self.config.parent_channel_id)) {
                    continue;
                }

                if thread.member.is_none() {
                    thread.id.join_thread(&ctx.http).await?;
                }

                log::info!("thread {} scheduled for load", thread.id);
                threads.insert(thread.id, std::sync::Arc::new(tokio::sync::Mutex::new(None)));
            }

            Ok::<_, anyhow::Error>(())
        })()
        .await
        {
            log::error!("error in guild_create: {:?}", e);
        }
    }

    async fn thread_create(&self, ctx: serenity::client::Context, thread: serenity::model::channel::GuildChannel) {
        if let Err(e) = (|| async {
            let me_id = self.me_id.lock().clone();

            if thread.parent_id != Some(serenity::model::id::ChannelId(self.config.parent_channel_id)) {
                return Ok(());
            }

            if thread.last_message_id != None {
                return Ok(());
            }

            thread.id.join_thread(&ctx.http).await?;
            if let Err(e) = thread.id.pin(&ctx.http, serenity::model::id::MessageId(thread.id.0)).await {
                log::warn!("could not pin first message: {:?}", e);
            }

            let thread_info = Thread::new(&ctx.http, me_id, thread.id).await?;
            log::info!("thread {} joined: {:?}", thread.id, thread_info.primary_message);

            let mut threads = self.threads.lock().await;
            threads.insert(thread.id, std::sync::Arc::new(tokio::sync::Mutex::new(Some(thread_info))));

            Ok::<_, anyhow::Error>(())
        })()
        .await
        {
            log::error!("error in thread_create: {:?}", e);
        }
    }

    async fn thread_update(&self, _ctx: serenity::client::Context, thread: serenity::model::channel::GuildChannel) {
        if let Err(e) = (|| async {
            if thread.parent_id != Some(serenity::model::id::ChannelId(self.config.parent_channel_id)) {
                return Ok(());
            }

            let mut threads = self.threads.lock().await;
            if thread.thread_metadata.unwrap().archived {
                log::info!("thread {} archived", thread.id);
                threads.remove(&thread.id);
            } else {
                match threads.entry(thread.id) {
                    std::collections::hash_map::Entry::Occupied(e) => {
                        let mut t = e.get().lock().await;
                        if let Some(t) = t.as_mut() {
                            t.mode = ThreadMode::from_channel_name(&thread.name);
                        }
                    }
                    std::collections::hash_map::Entry::Vacant(e) => {
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

    async fn thread_delete(&self, _ctx: serenity::client::Context, thread: serenity::model::channel::PartialGuildChannel) {
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

    async fn guild_member_update(&self, _ctx: serenity::client::Context, event: serenity::model::event::GuildMemberUpdateEvent) {
        if let Err(e) = (|| async {
            let mut resolver = self.resolver.lock().await;
            resolver.add_display_name(event.guild_id, event.user.id, event.nick.unwrap_or(event.user.name));
            Ok::<_, anyhow::Error>(())
        })()
        .await
        {
            log::error!("error in guild_member_update: {:?}", e);
        }
    }

    async fn message(&self, ctx: serenity::client::Context, new_message: serenity::model::channel::Message) {
        if let Err(e) = (|| async {
            let me_id = self.me_id.lock().clone();

            if new_message.kind != serenity::model::channel::MessageType::Regular
                && new_message.kind != serenity::model::channel::MessageType::InlineReply
            {
                return Ok(());
            }

            let thread = {
                let threads = self.threads.lock().await;
                let thread = if let Some(thread) = threads.get(&new_message.channel_id) {
                    thread
                } else {
                    return Ok(());
                };
                thread.clone()
            };

            let should_reply = new_message.author.id != me_id && new_message.mentions_user_id(me_id);
            let can_reply = thread.try_lock().is_ok();

            if should_reply && !can_reply {
                let content = new_message.content.clone();
                ctx.http.delete_message(new_message.channel_id.0, new_message.id.0).await?;
                new_message
                    .channel_id
                    .send_message(&ctx.http, |m| {
                        m.embed(|e| {
                            e.color(serenity::utils::colours::css::WARNING)
                                .description("I'm already replying, please wait for me to finish!")
                                .field("Original message", content, false)
                        })
                        .reference_message(&new_message)
                    })
                    .await?;
            }

            let mut thread = thread.lock().await;

            let thread = if let Some(thread) = thread.as_mut() {
                thread
            } else {
                log::info!("thread {} loaded", new_message.channel_id);
                *thread = Some(
                    Thread::new(&ctx.http, me_id, new_message.channel_id)
                        .await
                        .map_err(|e| anyhow::format_err!("Thread::new: {}", e))?,
                );
                thread.as_mut().unwrap()
            };

            while thread.messages.len() >= MAX_MESSAGE_BUFFER {
                thread.messages.pop_first();
            }
            thread.messages.insert(new_message.id, new_message.clone());

            if !should_reply || !can_reply {
                return Ok(());
            }

            new_message.channel_id.edit_thread(&ctx.http, |t| t.locked(true)).await?;

            let r = (|| async {
                let settings = ChatSettings::new(&thread.primary_message.content)?;

                let (input_tokens, messages) = {
                    let mut resolver = self.resolver.lock().await;

                    let system_message = openai::Message {
                        role: openai::Role::System,
                        name: None,
                        content: if thread.mode == ThreadMode::Multi {
                            format!(
                                "Your name is {}.\n\n{}\n\nDo not prefix your replies with your name and timestamp.",
                                resolver
                                    .resolve_display_name(&ctx.http, new_message.guild_id.unwrap(), me_id,)
                                    .await
                                    .map_err(|e| anyhow::format_err!("resolve_display_name: {}", e))?,
                                settings.system_message
                            )
                        } else {
                            settings.system_message.clone()
                        },
                    };

                    // every reply is primed with <im_start>assistant
                    let mut input_tokens = 2 + openai::count_message_tokens(&self.tokenizer, &system_message);

                    let mut messages = vec![];

                    for (_, message) in thread.messages.iter().rev() {
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
                            openai::Message {
                                role: openai::Role::User,
                                name: None,
                                content: match thread.mode {
                                    ThreadMode::Single => {
                                        if !message.mentions_user_id(me_id) {
                                            continue;
                                        }

                                        resolver
                                            .resolve_message(
                                                &ctx.http,
                                                new_message.guild_id.unwrap(),
                                                &STRIP_SINGLE_USER_REGEX.replace(&message.content, |c: &regex::Captures| {
                                                    if serenity::model::id::UserId(c["user_id"].parse::<u64>().unwrap()) == me_id {
                                                        "".to_string()
                                                    } else {
                                                        c[0].to_string()
                                                    }
                                                }),
                                            )
                                            .await
                                            .map_err(|e| anyhow::format_err!("resolve_message: {}", e))?
                                    }
                                    ThreadMode::Multi => format!(
                                        "{} at {} said:\n{}",
                                        resolver
                                            .resolve_display_name(&ctx.http, new_message.guild_id.unwrap(), message.author.id,)
                                            .await
                                            .map_err(|e| anyhow::format_err!("resolve_display_name: {}", e))?,
                                        new_message.timestamp.with_timezone(&chrono::Utc).to_rfc3339(),
                                        resolver
                                            .resolve_message(&ctx.http, new_message.guild_id.unwrap(), &message.content)
                                            .await
                                            .map_err(|e| anyhow::format_err!("resolve_message: {}", e))?
                                    ),
                                },
                            }
                        };

                        let message_tokens = openai::count_message_tokens(&self.tokenizer, &oai_message);

                        if input_tokens + message_tokens > self.config.max_input_tokens as usize {
                            break;
                        }

                        messages.push(oai_message);
                        input_tokens += message_tokens;
                    }

                    messages.push(system_message);
                    messages.reverse();

                    (input_tokens, messages)
                };

                let req = openai::ChatRequest {
                    messages,
                    model: "gpt-3.5-turbo".to_owned(),
                    temperature: settings.model_settings.temperature,
                    top_p: settings.model_settings.top_p,
                    frequency_penalty: settings.model_settings.frequency_penalty,
                    presence_penalty: settings.model_settings.presence_penalty,
                    max_tokens: Some(self.config.max_tokens - input_tokens as u32),
                };
                log::info!("{:#?}", req);

                let mut typing = Some(new_message.channel_id.start_typing(&ctx.http)?);

                let mut stream = Box::pin(
                    tokio::time::timeout(std::time::Duration::from_secs(30), self.chat_client.request(&req))
                        .await
                        .map_err(|e| anyhow::format_err!("timed out: {}", e))??,
                );

                let mut chunker = unichunk::Chunker::new(2000);
                while let Some(chunk) = tokio::time::timeout(std::time::Duration::from_secs(30), stream.next())
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
                        typing.take();
                        new_message
                            .channel_id
                            .send_message(&ctx.http, |m| m.content(&c).reference_message(&new_message))
                            .await
                            .map_err(|e| anyhow::format_err!("send_message: {}", e))?;
                        typing = Some(new_message.channel_id.start_typing(&ctx.http)?);
                    }
                }

                typing.take();

                let c = chunker.flush();
                if !c.is_empty() {
                    new_message
                        .channel_id
                        .send_message(&ctx.http, |m| m.content(&c).reference_message(&new_message))
                        .await
                        .map_err(|e| anyhow::format_err!("send_message: {}", e))?;
                }

                Ok::<_, anyhow::Error>(())
            })()
            .await;

            if let Err(e) = &r {
                new_message
                    .channel_id
                    .send_message(&ctx.http, |m| {
                        m.embed(|em| {
                            em.title("Error")
                                .color(serenity::utils::colours::css::DANGER)
                                .description(format!("{:?}", e))
                        })
                        .reference_message(&new_message)
                    })
                    .await
                    .map_err(|send_e| anyhow::format_err!("send error: {} ({})", send_e, e))?;
            }

            new_message.channel_id.edit_thread(&ctx.http, |t| t.locked(false)).await?;
            r
        })()
        .await
        {
            log::error!("error in message: {:?}", e);
        }
    }

    async fn message_update(&self, _ctx: serenity::client::Context, new_event: serenity::model::event::MessageUpdateEvent) {
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
        _deleted_message_id: serenity::model::id::MessageId,
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
            log::info!("thread {} flushed for message delete", channel_id);
            *thread = None;

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
        _multiple_deleted_messages_id: Vec<serenity::model::id::MessageId>,
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
            log::info!("thread {} flushed for message delete", channel_id);
            *thread = None;

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
    env_logger::builder().filter_module("peebot", log::LevelFilter::Info).init();

    log::info!("hello!");

    let opts = Opts::parse();

    let config = toml::from_str::<Config>(std::str::from_utf8(&std::fs::read(opts.config)?)?)?;

    let chat_client = openai::ChatClient::new(&config.openai_token);

    let intents = serenity::model::gateway::GatewayIntents::default()
        | serenity::model::gateway::GatewayIntents::MESSAGE_CONTENT
        | serenity::model::gateway::GatewayIntents::GUILD_MESSAGES
        | serenity::model::gateway::GatewayIntents::GUILDS
        | serenity::model::gateway::GatewayIntents::GUILD_MEMBERS;

    serenity::client::ClientBuilder::new(&config.discord_token, intents)
        .event_handler(Handler {
            resolver: tokio::sync::Mutex::new(Resolver::new()),
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
