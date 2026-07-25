package dev.sruti.di

import android.content.Context
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import androidx.room.Room
import dev.sruti.convert.ModelConverter
import dev.sruti.data.ChatDao
import dev.sruti.data.ChatDatabase
import dev.sruti.hub.CheckpointDownloader
import dev.sruti.hub.HuggingFaceApi
import dev.sruti.hub.ModelStore
import dev.sruti.settings.SettingsStore
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import okhttp3.OkHttpClient
import java.util.concurrent.TimeUnit
import javax.inject.Qualifier
import javax.inject.Singleton

@Qualifier
@Retention(AnnotationRetention.BINARY)
annotation class IoDispatcher

/** Threads that run inference and conversion, distinct from IO. */
@Qualifier
@Retention(AnnotationRetention.BINARY)
annotation class ComputeDispatcher

@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    @Provides
    @IoDispatcher
    fun provideIoDispatcher(): CoroutineDispatcher = Dispatchers.IO

    @Provides
    @ComputeDispatcher
    fun provideComputeDispatcher(): CoroutineDispatcher = Dispatchers.Default

    @Provides
    @Singleton
    fun provideOkHttpClient(): OkHttpClient = OkHttpClient.Builder()
        // Model shards are gigabytes over a phone connection. A read timeout
        // measured in seconds would abort perfectly healthy transfers; the
        // downloader's own cancellation is what stops a stuck one.
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(5, TimeUnit.MINUTES)
        .writeTimeout(5, TimeUnit.MINUTES)
        .retryOnConnectionFailure(true)
        .build()

    @Provides
    @Singleton
    fun provideSettingsStore(@ApplicationContext context: Context): SettingsStore =
        SettingsStore(context)

    @Provides
    @Singleton
    fun provideHuggingFaceApi(
        client: OkHttpClient,
        @IoDispatcher dispatcher: CoroutineDispatcher,
        settings: SettingsStore,
    ): HuggingFaceApi = HuggingFaceApi(client, dispatcher) { settings.currentToken() }

    @Provides
    @Singleton
    fun provideDownloader(
        client: OkHttpClient,
        api: HuggingFaceApi,
        @IoDispatcher dispatcher: CoroutineDispatcher,
    ): CheckpointDownloader = CheckpointDownloader(client, api, dispatcher)

    @Provides
    @Singleton
    fun provideModelStore(
        @ApplicationContext context: Context,
        @IoDispatcher dispatcher: CoroutineDispatcher,
    ): ModelStore = ModelStore(context, dispatcher)

    @Provides
    @Singleton
    fun provideModelConverter(
        @ComputeDispatcher dispatcher: CoroutineDispatcher,
    ): ModelConverter = ModelConverter(dispatcher)

    @Provides
    @Singleton
    fun provideChatDatabase(@ApplicationContext context: Context): ChatDatabase =
        Room.databaseBuilder(context, ChatDatabase::class.java, "sruti_chat.db").build()

    @Provides
    fun provideChatDao(database: ChatDatabase): ChatDao = database.chatDao()
}
