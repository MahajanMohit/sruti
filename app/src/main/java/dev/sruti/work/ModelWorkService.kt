package dev.sruti.work

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import dagger.hilt.android.AndroidEntryPoint
import dev.sruti.MainActivity
import dev.sruti.R
import dev.sruti.convert.CheckpointInfo
import dev.sruti.convert.ConvertEvent
import dev.sruti.convert.ModelConverter
import dev.sruti.convert.QuantType
import dev.sruti.hub.CheckpointDownloader
import dev.sruti.hub.DownloadEvent
import dev.sruti.hub.HuggingFaceApi
import dev.sruti.hub.LocalModel
import dev.sruti.hub.ModelStore
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * Runs the acquire-and-convert pipeline as a foreground service.
 *
 * Both halves are long: a 1-2B checkpoint is gigabytes to fetch and minutes to
 * convert. Android 16 tightens JobScheduler quotas even when a foreground service
 * is running, and a plain coroutine tied to an Activity would be killed the moment
 * the user switches away — which, given the duration, they certainly will.
 */
@AndroidEntryPoint
class ModelWorkService : Service() {

    @Inject lateinit var api: HuggingFaceApi
    @Inject lateinit var downloader: CheckpointDownloader
    @Inject lateinit var converter: ModelConverter
    @Inject lateinit var store: ModelStore

    private val scope = CoroutineScope(SupervisorJob())
    private var job: Job? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_CANCEL -> {
                job?.cancel()
                stopSelfCleanly()
                return START_NOT_STICKY
            }
        }

        val repoId = intent?.getStringExtra(EXTRA_REPO_ID)
        val quantType = QuantType.fromId(intent?.getStringExtra(EXTRA_QUANT) ?: QuantType.Q4_K_M.id)
        if (repoId.isNullOrBlank()) {
            stopSelfCleanly()
            return START_NOT_STICKY
        }

        if (job?.isActive == true) {
            // One conversion at a time: two at once would fight over CPU, disk and
            // thermal budget and finish slower than running them in sequence.
            return START_NOT_STICKY
        }

        startForeground(
            NOTIFICATION_ID,
            buildNotification("Preparing", repoId, null),
            foregroundServiceType(),
        )

        job = scope.launch { runPipeline(repoId, quantType) }
        return START_NOT_STICKY
    }

    private suspend fun runPipeline(repoId: String, quantType: QuantType) {
        fun publish(state: ModelJobState) {
            _state.value = state
            if (state is ModelJobState.Running) {
                notify(state)
            }
        }

        try {
            publish(
                ModelJobState.Running(repoId, quantType, ModelJobState.Running.Phase.Resolving),
            )

            // config.json is a few kilobytes. Checking it first is what stops a
            // 2.5 GB download of something that cannot be converted.
            val info = CheckpointInfo.inspect(api.fetchText(repoId, "config.json"))
            if (!info.supported) {
                publish(ModelJobState.Failed(repoId, info.error))
                return
            }

            val checkpoint = api.listFiles(repoId)
            if (!checkpoint.hasSafetensors) {
                publish(
                    ModelJobState.Failed(
                        repoId,
                        "$repoId has no .safetensors weights. Sruti converts safetensors " +
                            "checkpoints, not pre-built GGUF repositories.",
                    ),
                )
                return
            }

            val checkpointDir = store.checkpointDir(repoId)
            val outputFile = store.outputFileFor(repoId, quantType)

            // Leave room for the conversion that follows: the F16 intermediate and
            // the quantized output both exist on disk at once.
            val reserve = converter.estimatedWorkingBytes(checkpoint.totalBytes, quantType)

            downloader.download(checkpoint, checkpointDir, reserveBytes = reserve)
                .collect { event ->
                    when (event) {
                        is DownloadEvent.Progress -> publish(
                            ModelJobState.Running(
                                repoId = repoId,
                                quantType = quantType,
                                phase = ModelJobState.Running.Phase.Downloading,
                                detail = event.currentFile,
                                fraction = event.fraction,
                                bytesDone = event.bytesDone,
                                bytesTotal = event.bytesTotal,
                            ),
                        )
                        is DownloadEvent.Skipped, is DownloadEvent.Completed -> Unit
                    }
                }

            val warnings = mutableListOf<String>()
            converter.convert(
                modelDir = checkpointDir,
                outputFile = outputFile,
                quantType = quantType,
                modelName = repoId.substringAfterLast('/'),
            ).collect { event ->
                when (event) {
                    is ConvertEvent.Progress -> publish(
                        ModelJobState.Running(
                            repoId = repoId,
                            quantType = quantType,
                            phase = ModelJobState.Running.Phase.Converting,
                            detail = event.stage.label,
                            fraction = event.fraction,
                        ),
                    )
                    is ConvertEvent.Warning -> warnings += event.message
                    is ConvertEvent.Completed -> Unit
                }
            }

            store.writeMetadata(
                outputFile,
                LocalModel(
                    fileName = outputFile.name,
                    displayName = repoId.substringAfterLast('/'),
                    sourceRepo = repoId,
                    architecture = info.displayName,
                    quantType = quantType.id,
                    parameterCount = info.parameterCount,
                    convertedAtMillis = System.currentTimeMillis(),
                ),
            )

            // The checkpoint is several times the size of the result and is not
            // needed again. Keeping it would fill the device for no benefit.
            publish(
                ModelJobState.Running(repoId, quantType, ModelJobState.Running.Phase.Cleaning),
            )
            checkpointDir.deleteRecursively()

            _state.value = ModelJobState.Succeeded(repoId, outputFile, warnings)
            notifyFinished("Ready: ${outputFile.nameWithoutExtension}")
        } catch (t: Throwable) {
            if (t is kotlinx.coroutines.CancellationException) {
                _state.value = ModelJobState.Idle
                throw t
            }
            _state.value = ModelJobState.Failed(repoId, t.message ?: "conversion failed")
            notifyFinished("Failed: ${t.message ?: "unknown error"}")
        } finally {
            stopSelfCleanly()
        }
    }

    private fun stopSelfCleanly() {
        stopForeground(STOP_FOREGROUND_DETACH)
        stopSelf()
    }

    override fun onDestroy() {
        scope.cancel()
        super.onDestroy()
    }

    // --- notifications ------------------------------------------------------

    private fun foregroundServiceType(): Int =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
        } else {
            0
        }

    private fun createNotificationChannel() {
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                "Model preparation",
                NotificationManager.IMPORTANCE_LOW,
            ).apply {
                description = "Downloading and converting models"
                setShowBadge(false)
            },
        )
    }

    private fun notify(state: ModelJobState.Running) {
        val manager = getSystemService(NotificationManager::class.java)
        manager.notify(
            NOTIFICATION_ID,
            buildNotification(state.phase.label, state.repoId, state.fraction),
        )
    }

    private fun notifyFinished(text: String) {
        val manager = getSystemService(NotificationManager::class.java)
        manager.notify(
            NOTIFICATION_ID,
            NotificationCompat.Builder(this, CHANNEL_ID)
                .setContentTitle("Sruti")
                .setContentText(text)
                .setSmallIcon(R.drawable.ic_notification)
                .setContentIntent(contentIntent())
                .setAutoCancel(true)
                .setOngoing(false)
                .build(),
        )
    }

    private fun buildNotification(title: String, repoId: String, fraction: Float?): Notification =
        NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(title)
            .setContentText(repoId)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentIntent(contentIntent())
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .apply {
                if (fraction != null) {
                    setProgress(100, (fraction * 100).toInt().coerceIn(0, 100), false)
                } else {
                    setProgress(0, 0, true)
                }
            }
            .addAction(
                0,
                "Cancel",
                PendingIntent.getService(
                    this@ModelWorkService,
                    1,
                    Intent(this@ModelWorkService, ModelWorkService::class.java)
                        .setAction(ACTION_CANCEL),
                    PendingIntent.FLAG_IMMUTABLE,
                ),
            )
            .build()

    private fun contentIntent(): PendingIntent = PendingIntent.getActivity(
        this,
        0,
        Intent(this, MainActivity::class.java),
        PendingIntent.FLAG_IMMUTABLE,
    )

    companion object {
        private const val CHANNEL_ID = "model_work"
        private const val NOTIFICATION_ID = 1001
        private const val ACTION_CANCEL = "dev.sruti.action.CANCEL_MODEL_WORK"
        private const val EXTRA_REPO_ID = "repo_id"
        private const val EXTRA_QUANT = "quant"

        private val _state = MutableStateFlow<ModelJobState>(ModelJobState.Idle)

        /**
         * Observable pipeline state.
         *
         * Held statically rather than bound through the service connection so the
         * UI can observe a job that started before this Activity existed — the
         * normal case, since these jobs outlive the screen.
         */
        val state: StateFlow<ModelJobState> = _state.asStateFlow()

        fun start(context: Context, repoId: String, quantType: QuantType) {
            context.startForegroundService(
                Intent(context, ModelWorkService::class.java)
                    .putExtra(EXTRA_REPO_ID, repoId)
                    .putExtra(EXTRA_QUANT, quantType.id),
            )
        }

        fun cancel(context: Context) {
            context.startService(
                Intent(context, ModelWorkService::class.java).setAction(ACTION_CANCEL),
            )
        }

        fun acknowledge() {
            _state.value = ModelJobState.Idle
        }
    }
}
