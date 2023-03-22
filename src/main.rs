mod backend;
mod openai;
mod unichunk;

use clap::Parser;
use futures_util::StreamExt;

#[derive(Debug, PartialEq)]
enum ThreadMode {
    Single,
    Multi,
}

#[derive(Debug)]
struct ChatSettings {
    system_message: String,
    parameters: toml::Value,
}

static FORGET_EMOJI: &str = "❌";

impl ChatSettings {
    fn new(s: &str) -> Result<Self, anyhow::Error> {
        static STRIP_TRAILING_WHITESPACE_REGEX: once_cell::sync::Lazy<regex::Regex> =
            once_cell::sync::Lazy::new(|| regex::Regex::new(r"[ \t]+\n").unwrap());

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
            parameters: parts[1].map_or_else(|| Ok(toml::Table::new().into()), |v| toml::from_str::<toml::Value>(v))?,
        })
    }
}

#[derive(Debug)]
struct ThreadInfo {
    primary_message: serenity::model::channel::Message,
    messages: std::collections::BTreeMap<serenity::model::id::MessageId, serenity::model::channel::Message>,
    mode: ThreadMode,
    backend: Option<String>,
}

impl ThreadInfo {
    async fn new(
        http: impl AsRef<serenity::http::Http>,
        id: serenity::model::id::ChannelId,
        tags: &std::collections::HashMap<serenity::model::id::ForumTagId, String>,
        message_history_size: usize,
    ) -> Result<Self, serenity::Error> {
        let primary_message = id.message(&http, id.0).await?;
        let mut messages = std::collections::BTreeMap::new();

        let mut messages_it = Box::pin(id.messages_iter(&http)).take(message_history_size);
        while let Some(message) = messages_it.next().await {
            let message = message?;
            if message.id.0 == id.0 {
                break;
            }
            messages.insert(message.id, message);
        }

        let channel = if let serenity::model::prelude::Channel::Guild(guild_channel) = http.as_ref().get_channel(id.0).await? {
            guild_channel
        } else {
            unreachable!();
        };

        let mut ti = Self {
            primary_message,
            messages,
            mode: ThreadMode::Single,
            backend: None,
        };

        ti.update_from_tags(&channel, &tags);

        Ok(ti)
    }

    fn update_from_tags(
        &mut self,
        thread: &serenity::model::channel::GuildChannel,
        tags: &std::collections::HashMap<serenity::model::id::ForumTagId, String>,
    ) {
        self.mode = ThreadMode::Single;
        self.backend = None;

        for tag in thread.applied_tags.iter() {
            let tag_name = if let Some(tag_name) = tags.get(&tag) {
                tag_name
            } else {
                continue;
            };

            if tag_name == "multi" {
                self.mode = ThreadMode::Multi;
            } else if let Some(backend_name) = tag_name.strip_prefix("use ") {
                self.backend = Some(backend_name.to_string());
            }
        }
    }
}

struct Resolver {
    display_names: lru::LruCache<(serenity::model::id::GuildId, serenity::model::id::UserId), String>,
}

impl Resolver {
    fn new(cache_size: usize) -> Self {
        Self {
            display_names: lru::LruCache::new(std::num::NonZeroUsize::new(cache_size).unwrap()),
        }
    }

    fn hint_display_name(&mut self, guild_id: serenity::model::id::GuildId, user_id: serenity::model::id::UserId, name: String) {
        if !self.display_names.contains(&(guild_id, user_id)) {
            // If we don't have the display name cached, don't add it.
            return;
        }
        self.display_names.put((guild_id, user_id), name);
    }

    async fn resolve_display_name(
        &mut self,
        http: impl AsRef<serenity::http::Http>,
        guild_id: serenity::model::id::GuildId,
        user_id: serenity::model::id::UserId,
    ) -> Result<&str, serenity::Error> {
        if self.display_names.get(&(guild_id, user_id)).is_none() {
            let member = http.as_ref().get_member(guild_id.0, user_id.0).await?;
            self.display_names.put((guild_id, user_id), member.display_name().into_owned());
        }
        Ok(self.display_names.get(&(guild_id, user_id)).unwrap())
    }

    async fn resolve_message(
        &mut self,
        http: impl AsRef<serenity::http::Http>,
        guild_id: serenity::model::id::GuildId,
        content: &str,
    ) -> Result<String, serenity::Error> {
        let mut s = String::new();
        let mut last_index = 0;

        static RESOLVE_MESSAGE_REGEX: once_cell::sync::Lazy<regex::Regex> =
            once_cell::sync::Lazy::new(|| regex::Regex::new(r"<@!?(?P<user_id>\d+)>|<a?:(?P<emoji_name>\w+):\d+>|<#(?P<channel_id>\d+)>").unwrap());

        for capture in RESOLVE_MESSAGE_REGEX.captures_iter(content) {
            let m = capture.get(0).unwrap();

            s.push_str(&content[last_index..m.start()]);

            let repl = if let Some(subm) = capture.name("user_id") {
                let user_id = subm.as_str().parse::<u64>().unwrap();
                self.resolve_display_name(&http, guild_id, user_id.into()).await?.to_string()
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
    config: Config,
    parent_channel_id: serenity::model::id::ChannelId,
    backends: indexmap::IndexMap<String, Box<dyn backend::Backend + Send + Sync>>,
    thread_cache: tokio::sync::Mutex<ThreadCache>,
    tags: tokio::sync::Mutex<std::collections::HashMap<serenity::model::id::ForumTagId, String>>,
}

struct ThreadCache {
    ids: std::collections::HashSet<serenity::model::id::ChannelId>,
    infos: lru::LruCache<serenity::model::id::ChannelId, std::sync::Arc<tokio::sync::Mutex<ThreadInfo>>>,
}

impl ThreadCache {
    fn new(cache_size: usize) -> Self {
        Self {
            ids: std::collections::HashSet::new(),
            infos: lru::LruCache::new(std::num::NonZeroUsize::new(cache_size).unwrap()),
        }
    }

    fn add(&mut self, thread_id: serenity::model::id::ChannelId) {
        self.ids.insert(thread_id);
    }

    fn remove(&mut self, thread_id: serenity::model::id::ChannelId) {
        self.ids.remove(&thread_id);
        self.infos.pop(&thread_id);
    }

    fn get(&mut self, thread_id: serenity::model::id::ChannelId) -> Option<std::sync::Arc<tokio::sync::Mutex<ThreadInfo>>> {
        self.infos.get(&thread_id).cloned()
    }

    async fn load(
        &mut self,
        http: impl AsRef<serenity::http::Http>,
        thread_id: serenity::model::id::ChannelId,
        tags: &std::collections::HashMap<serenity::model::id::ForumTagId, String>,
        message_history_size: usize,
    ) -> Result<Option<std::sync::Arc<tokio::sync::Mutex<ThreadInfo>>>, serenity::Error> {
        if !self.ids.contains(&thread_id) {
            return Ok(None);
        }

        if let Some(info) = self.infos.get(&thread_id) {
            return Ok(Some(info.clone()));
        }

        let thread_info = std::sync::Arc::new(tokio::sync::Mutex::new(
            ThreadInfo::new(http, thread_id, tags, message_history_size).await?,
        ));
        self.infos.put(thread_id, thread_info.clone());
        Ok(Some(thread_info))
    }
}

static STRIP_SINGLE_USER_REGEX: once_cell::sync::Lazy<regex::Regex> =
    once_cell::sync::Lazy::new(|| regex::Regex::new(r"^\s*<@!?(?P<user_id>\d+)>\s*").unwrap());

const FORGET_COMMAND_NAME: &str = "forget";
const INJECT_COMMAND_NAME: &str = "inject";
const INJECT_SYSTEM_COMMAND_NAME: &str = "injectsystem";

#[async_trait::async_trait]
impl serenity::client::EventHandler for Handler {
    async fn ready(&self, ctx: serenity::client::Context, data_about_bot: serenity::model::gateway::Ready) {
        if let Err(e) = (|| async {
            *self.me_id.lock() = data_about_bot.user.id;

            serenity::model::application::command::Command::set_global_application_commands(&ctx.http, |cmds| {
                cmds.create_application_command(|c| {
                    c.name(FORGET_COMMAND_NAME)
                        .description("Add a break in the chat log to forget everything before it.")
                })
                .create_application_command(|c| {
                    c.name(INJECT_COMMAND_NAME)
                        .description("Just make me say something directly.")
                        .create_option(|o| {
                            o.name("content")
                                .description("The text to say.")
                                .kind(serenity::model::application::command::CommandOptionType::String)
                                .required(true)
                        })
                })
                .create_application_command(|c| {
                    c.name(INJECT_SYSTEM_COMMAND_NAME)
                        .description("Inject a new system message.")
                        .create_option(|o| {
                            o.name("content")
                                .description("The text to say.")
                                .kind(serenity::model::application::command::CommandOptionType::String)
                                .required(true)
                        })
                })
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

            match app_command.kind {
                serenity::model::application::interaction::InteractionType::ApplicationCommand => match app_command.data.name.as_str() {
                    FORGET_COMMAND_NAME => {
                        app_command
                            .create_interaction_response(&ctx.http, |r| {
                                r.interaction_response_data(|d| {
                                    d.embed(|e| {
                                        e.color(serenity::utils::colours::css::POSITIVE).description(
                                            "Okay, forgetting everything from here. If you want me to remember, just delete this message.",
                                        )
                                    })
                                })
                            })
                            .await?;
                    }
                    INJECT_COMMAND_NAME => {
                        let content = if let Some(content) = app_command.data.options.get(0).and_then(|v| v.value.as_ref()).and_then(|v| v.as_str()) {
                            content
                        } else {
                            return Ok(());
                        };
                        app_command
                            .create_interaction_response(&ctx.http, |r| r.interaction_response_data(|d| d.content(content)))
                            .await?;
                    }
                    INJECT_SYSTEM_COMMAND_NAME => {
                        let content = if let Some(content) = app_command.data.options.get(0).and_then(|v| v.value.as_ref()).and_then(|v| v.as_str()) {
                            content
                        } else {
                            return Ok(());
                        };
                        app_command
                            .create_interaction_response(&ctx.http, |r| r.interaction_response_data(|d| d.content(content)))
                            .await?;
                    }
                    _ => {}
                },
                _ => {}
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
            let mut thread_cache = self.thread_cache.lock().await;
            for thread in guild.threads.iter() {
                if !thread.parent_id.map(|thread_id| self.parent_channel_id == thread_id).unwrap_or(false) {
                    continue;
                }

                if thread.member.is_none() {
                    thread.id.join_thread(&ctx.http).await?;
                }

                log::info!("thread {} scheduled for load", thread.id);
                thread_cache.add(thread.id);
            }

            let parent_channel = if let serenity::model::channel::Channel::Guild(guild_channel) = &guild.channels[&self.parent_channel_id] {
                guild_channel
            } else {
                return Ok(());
            };

            let mut tags = self.tags.lock().await;
            *tags = parent_channel
                .available_tags
                .iter()
                .map(|tag| (tag.id, tag.name.clone()))
                .collect::<std::collections::HashMap<_, _>>();

            Ok::<_, anyhow::Error>(())
        })()
        .await
        {
            log::error!("error in guild_create: {:?}", e);
        }
    }

    async fn channel_update(&self, _ctx: serenity::client::Context, channel: serenity::model::channel::Channel) {
        if let Err(e) = (|| async {
            let channel = if let serenity::model::channel::Channel::Guild(guild_channel) = channel {
                guild_channel
            } else {
                return Ok(());
            };

            if channel.id != self.parent_channel_id {
                return Ok(());
            }

            let mut tags = self.tags.lock().await;
            *tags = channel
                .available_tags
                .iter()
                .map(|tag| (tag.id, tag.name.clone()))
                .collect::<std::collections::HashMap<_, _>>();

            Ok::<_, anyhow::Error>(())
        })()
        .await
        {
            log::error!("error in channel_update: {:?}", e);
        }
    }

    async fn thread_create(&self, ctx: serenity::client::Context, thread: serenity::model::channel::GuildChannel) {
        if let Err(e) = (|| async {
            if !thread.parent_id.map(|thread_id| self.parent_channel_id == thread_id).unwrap_or(false) {
                return Ok(());
            }

            if thread.last_message_id != None {
                return Ok(());
            }

            thread.id.join_thread(&ctx.http).await?;
            if let Err(e) = thread.id.pin(&ctx.http, serenity::model::id::MessageId(thread.id.0)).await {
                log::warn!("could not pin first message: {:?}", e);
            }

            let mut thread_cache = self.thread_cache.lock().await;
            thread_cache.add(thread.id);

            // Optimization only, not strictly required.
            let tags = self.tags.lock().await;
            thread_cache.load(&ctx.http, thread.id, &*tags, self.config.message_history_size).await?;

            Ok::<_, anyhow::Error>(())
        })()
        .await
        {
            log::error!("error in thread_create: {:?}", e);
        }
    }

    async fn thread_update(&self, _ctx: serenity::client::Context, thread: serenity::model::channel::GuildChannel) {
        if let Err(e) = (|| async {
            if !thread.parent_id.map(|thread_id| self.parent_channel_id == thread_id).unwrap_or(false) {
                return Ok(());
            }

            let mut thread_cache = self.thread_cache.lock().await;
            if thread.thread_metadata.unwrap().archived {
                log::info!("thread {} archived", thread.id);
                thread_cache.remove(thread.id);
            } else {
                thread_cache.add(thread.id);
                if let Some(t) = thread_cache.get(thread.id) {
                    let mut t = t.lock().await;
                    let tags = self.tags.lock().await;
                    t.update_from_tags(&thread, &*tags);
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
            let mut thread_cache = self.thread_cache.lock().await;
            log::info!("thread {} deleted", thread.id);
            thread_cache.remove(thread.id);
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
            resolver.hint_display_name(event.guild_id, event.user.id, event.nick.unwrap_or(event.user.name));
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

            let thread = {
                let mut thread_cache = self.thread_cache.lock().await;
                let tags = self.tags.lock().await;
                let thread = if let Some(thread) = thread_cache
                    .load(&ctx.http, new_message.channel_id, &*tags, self.config.message_history_size)
                    .await?
                {
                    thread
                } else {
                    return Ok(());
                };
                thread
            };

            let should_reply = new_message.author.id != me_id
                && new_message.mentions_user_id(me_id)
                && (new_message.kind == serenity::model::channel::MessageType::Regular
                    || new_message.kind == serenity::model::channel::MessageType::InlineReply);

            let can_reply = thread.try_lock().is_ok();

            if should_reply && !can_reply {
                ctx.http.delete_message(new_message.channel_id.0, new_message.id.0).await?;
                new_message
                    .channel_id
                    .send_message(&ctx.http, |m| {
                        m.embed(|e| {
                            e.color(serenity::utils::colours::css::WARNING)
                                .description("I'm already replying, please wait for me to finish!")
                                .field("Original message", format!("```\n{}\n```", new_message.content), false)
                                .footer(|f| {
                                    f.icon_url(
                                        new_message
                                            .author
                                            .static_avatar_url()
                                            .unwrap_or_else(|| new_message.author.default_avatar_url()),
                                    )
                                    .text(format!("{}#{:04}", new_message.author.name, new_message.author.discriminator))
                                })
                                .timestamp(new_message.timestamp)
                        })
                    })
                    .await?;
                return Ok(());
            }

            let mut thread = thread.lock().await;

            while thread.messages.len() >= self.config.message_history_size {
                thread.messages.pop_first();
            }
            thread.messages.insert(new_message.id, new_message.clone());

            if !should_reply {
                return Ok(());
            }

            let settings = ChatSettings::new(&thread.primary_message.content)?;

            let (backend_name, backend) = if let Some((backend_name, backend)) = thread
                .backend
                .as_ref()
                .and_then(|backend_name| self.backends.get(backend_name).map(|backend| (backend_name, backend)))
                .or_else(|| self.backends.first())
            {
                (backend_name, backend)
            } else {
                return Ok(());
            };

            let r = (|| async {
                let messages = {
                    let mut resolver = self.resolver.lock().await;

                    let system_message = backend::Message {
                        role: backend::Role::System,
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

                    let mut input_tokens = backend.num_overhead_tokens() + backend.count_message_tokens(&system_message);

                    let mut messages = vec![];

                    for (_, message) in thread.messages.iter().rev() {
                        if message.author.id == me_id
                            && message
                                .interaction
                                .as_ref()
                                .map(|i| {
                                    i.kind == serenity::model::application::interaction::InteractionType::ApplicationCommand
                                        && i.name == FORGET_COMMAND_NAME
                                })
                                .unwrap_or(false)
                        {
                            break;
                        }

                        if message.content.is_empty() {
                            continue;
                        }

                        if message.kind != serenity::model::channel::MessageType::Regular
                            && message.kind != serenity::model::channel::MessageType::InlineReply
                            && message.kind != serenity::model::channel::MessageType::ChatInputCommand
                        {
                            continue;
                        }

                        if message
                            .reactions
                            .iter()
                            .any(|r| r.reaction_type == serenity::model::channel::ReactionType::Unicode(FORGET_EMOJI.to_string()))
                        {
                            continue;
                        }

                        let oai_message = if message.author.id == me_id {
                            backend::Message {
                                role: if message
                                    .interaction
                                    .as_ref()
                                    .map(|i| {
                                        i.kind == serenity::model::application::interaction::InteractionType::ApplicationCommand
                                            && i.name == INJECT_SYSTEM_COMMAND_NAME
                                    })
                                    .unwrap_or(false)
                                {
                                    backend::Role::System
                                } else {
                                    backend::Role::Assistant
                                },
                                name: None,
                                content: message.content.clone(),
                            }
                        } else {
                            backend::Message {
                                role: backend::Role::User,
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
                                            .resolve_display_name(&ctx.http, new_message.guild_id.unwrap(), message.author.id)
                                            .await
                                            .map_err(|e| anyhow::format_err!("resolve_display_name: {}", e))?
                                            .to_owned(),
                                        new_message.timestamp.with_timezone(&chrono::Utc).to_rfc3339(),
                                        resolver
                                            .resolve_message(&ctx.http, new_message.guild_id.unwrap(), &message.content)
                                            .await
                                            .map_err(|e| anyhow::format_err!("resolve_message: {}", e))?
                                            .to_owned()
                                    ),
                                },
                            }
                        };

                        let message_tokens = backend.count_message_tokens(&oai_message);

                        if input_tokens + message_tokens > self.config.max_input_tokens as usize {
                            break;
                        }

                        messages.push(oai_message);
                        input_tokens += message_tokens;
                    }

                    messages.push(system_message);
                    messages.reverse();

                    messages
                };

                log::info!("{} ({:?}) <- {:#?}", backend_name, settings.parameters, messages);

                let mut typing = Some(new_message.channel_id.start_typing(&ctx.http)?);

                let mut stream = tokio::time::timeout(backend.request_timeout(), backend.request(&messages, &settings.parameters))
                    .await
                    .map_err(|e| anyhow::format_err!("timed out: {}", e))??;

                let mut chunker = unichunk::Chunker::new(2000);
                while let Some(content) = tokio::time::timeout(backend.chunk_timeout(), stream.next())
                    .await
                    .map_err(|e| anyhow::format_err!("timed out: {}", e))?
                {
                    let content = content?;

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
                                .field("Original message", format!("```\n{}\n```", new_message.content), false)
                                .footer(|f| {
                                    f.icon_url(
                                        new_message
                                            .author
                                            .static_avatar_url()
                                            .unwrap_or_else(|| new_message.author.default_avatar_url()),
                                    )
                                    .text(format!("{}#{:04}", new_message.author.name, new_message.author.discriminator))
                                })
                        })
                    })
                    .await
                    .map_err(|send_e| anyhow::format_err!("send error: {} ({})", send_e, e))?;
                ctx.http.delete_message(new_message.channel_id.0, new_message.id.0).await?;
            }

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
                let mut thread_cache = self.thread_cache.lock().await;
                let thread = if let Some(thread) = thread_cache.get(new_event.channel_id) {
                    thread
                } else {
                    // If the thread is not loaded, just ignore it.
                    return Ok(());
                };
                thread
            };

            let mut thread = thread.lock().await;
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

    async fn reaction_add(&self, _ctx: serenity::client::Context, reaction: serenity::model::channel::Reaction) {
        if let Err(e) = (|| async {
            let me_id = self.me_id.lock().clone();

            let thread = {
                let mut thread_cache = self.thread_cache.lock().await;
                let thread = if let Some(thread) = thread_cache.get(reaction.channel_id) {
                    thread
                } else {
                    // If the thread is not loaded, just ignore it.
                    return Ok(());
                };
                thread
            };

            let mut thread = thread.lock().await;
            let message = if let Some(message) = thread.messages.get_mut(&reaction.message_id) {
                message
            } else {
                return Ok(());
            };

            let message_reaction = if let Some(message_reaction) = message.reactions.iter_mut().find(|r| r.reaction_type == reaction.emoji) {
                message_reaction
            } else {
                // Kind of janky, but whatever.
                static EMPTY_MESSAGE_REACTION: once_cell::sync::Lazy<serenity::model::channel::MessageReaction> =
                    once_cell::sync::Lazy::new(|| serde_json::from_str("{\"count\": 0, \"me\": false, \"emoji\": {\"name\": \"\"}}").unwrap());

                let mut message_reaction = EMPTY_MESSAGE_REACTION.clone();
                message_reaction.count = 0;
                message_reaction.me = reaction
                    .member
                    .and_then(|member| member.user.map(|user| user.id == me_id))
                    .unwrap_or(false);
                message_reaction.reaction_type = reaction.emoji;

                message.reactions.push(message_reaction);
                message.reactions.last_mut().unwrap()
            };
            message_reaction.count += 1;

            Ok::<_, anyhow::Error>(())
        })()
        .await
        {
            log::error!("error in reaction_remove_all: {:?}", e);
        }
    }

    async fn reaction_remove(&self, _ctx: serenity::client::Context, reaction: serenity::model::channel::Reaction) {
        if let Err(e) = (|| async {
            let me_id = self.me_id.lock().clone();

            let thread = {
                let mut thread_cache = self.thread_cache.lock().await;
                let thread = if let Some(thread) = thread_cache.get(reaction.channel_id) {
                    thread
                } else {
                    // If the thread is not loaded, just ignore it.
                    return Ok(());
                };
                thread
            };

            let mut thread = thread.lock().await;
            let message = if let Some(message) = thread.messages.get_mut(&reaction.message_id) {
                message
            } else {
                return Ok(());
            };

            message.reactions = message
                .reactions
                .iter()
                .map(|r| {
                    let mut r = r.clone();

                    if r.reaction_type != reaction.emoji {
                        return r;
                    }

                    r.count -= 1;
                    if reaction
                        .member
                        .as_ref()
                        .and_then(|member| member.user.as_ref().map(|user| user.id == me_id))
                        .unwrap_or(false)
                    {
                        r.me = false;
                    }

                    r
                })
                .filter(|r| r.count > 0)
                .collect();

            Ok::<_, anyhow::Error>(())
        })()
        .await
        {
            log::error!("error in reaction_remove_all: {:?}", e);
        }
    }

    async fn reaction_remove_all(
        &self,
        _ctx: serenity::client::Context,
        channel_id: serenity::model::id::ChannelId,
        message_id: serenity::model::id::MessageId,
    ) {
        if let Err(e) = (|| async {
            let thread = {
                let mut thread_cache = self.thread_cache.lock().await;
                let thread = if let Some(thread) = thread_cache.get(channel_id) {
                    thread
                } else {
                    // If the thread is not loaded, just ignore it.
                    return Ok(());
                };
                thread
            };

            let mut thread = thread.lock().await;
            let message = if let Some(message) = thread.messages.get_mut(&message_id) {
                message
            } else {
                return Ok(());
            };

            message.reactions.clear();

            Ok::<_, anyhow::Error>(())
        })()
        .await
        {
            log::error!("error in reaction_remove_all: {:?}", e);
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
                let mut thread_cache = self.thread_cache.lock().await;
                let thread = if let Some(thread) = thread_cache.get(channel_id) {
                    thread
                } else {
                    // If the thread is not loaded, just ignore it.
                    return Ok(());
                };
                thread
            };

            let mut thread = thread.lock().await;
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
                let mut thread_cache = self.thread_cache.lock().await;
                let thread = if let Some(thread) = thread_cache.get(channel_id) {
                    thread
                } else {
                    // If the thread is not loaded, just ignore it.
                    return Ok(());
                };
                thread
            };

            let mut thread = thread.lock().await;
            for deleted_message_id in multiple_deleted_messages_id {
                thread.messages.remove(&deleted_message_id);
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

const fn display_name_resolver_cache_size_default() -> usize {
    2000
}

const fn thread_cache_size_default() -> usize {
    2000
}

const fn message_history_size_default() -> usize {
    2000
}

#[derive(serde::Deserialize)]
struct Config {
    backends: indexmap::IndexMap<String, toml::Value>,

    discord_token: String,

    parent_channel_id: u64,

    #[serde(default = "max_input_tokens_default")]
    max_input_tokens: u32,

    #[serde(default = "display_name_resolver_cache_size_default")]
    display_name_resolver_cache_size: usize,

    #[serde(default = "thread_cache_size_default")]
    thread_cache_size: usize,

    #[serde(default = "message_history_size_default")]
    message_history_size: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::builder().filter_module("peebot", log::LevelFilter::Info).init();

    log::info!("hello!");

    let opts = Opts::parse();

    let config = toml::from_str::<Config>(std::str::from_utf8(&std::fs::read(opts.config)?)?)?;

    let mut backends: indexmap::IndexMap<String, Box<dyn backend::Backend + Sync + Send>> = indexmap::IndexMap::new();
    for (name, c) in config.backends.iter() {
        backends.insert(
            name.clone(),
            backend::new_backend_from_config(c.get("type").unwrap().as_str().unwrap().to_string(), c.clone())?,
        );
    }

    let intents = serenity::model::gateway::GatewayIntents::default()
        | serenity::model::gateway::GatewayIntents::MESSAGE_CONTENT
        | serenity::model::gateway::GatewayIntents::GUILD_MESSAGES
        | serenity::model::gateway::GatewayIntents::GUILD_MESSAGE_REACTIONS
        | serenity::model::gateway::GatewayIntents::GUILDS
        | serenity::model::gateway::GatewayIntents::GUILD_MEMBERS;

    let resolver = tokio::sync::Mutex::new(Resolver::new(config.display_name_resolver_cache_size));
    let thread_cache = tokio::sync::Mutex::new(ThreadCache::new(config.thread_cache_size));

    serenity::client::ClientBuilder::new(&config.discord_token, intents)
        .event_handler(Handler {
            resolver,
            me_id: parking_lot::Mutex::new(serenity::model::id::UserId::default()),
            parent_channel_id: serenity::model::id::ChannelId(config.parent_channel_id),
            tags: tokio::sync::Mutex::new(std::collections::HashMap::new()),
            config,
            backends,
            thread_cache,
        })
        .await?
        .start()
        .await?;

    Ok(())
}
