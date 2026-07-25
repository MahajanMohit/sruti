package dev.sruti.data

import androidx.room.Dao
import androidx.room.Database
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.Insert
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.RoomDatabase
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

@Entity(tableName = "conversations")
data class ConversationEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val title: String,
    /** Which model produced this conversation; its cache is not portable. */
    val modelFileName: String,
    val systemPrompt: String = "",
    val createdAtMillis: Long = System.currentTimeMillis(),
    val updatedAtMillis: Long = System.currentTimeMillis(),
)

@Entity(
    tableName = "messages",
    foreignKeys = [
        ForeignKey(
            entity = ConversationEntity::class,
            parentColumns = ["id"],
            childColumns = ["conversationId"],
            // Deleting a conversation must not leave its messages behind; doing
            // this in SQL rather than in Kotlin keeps it true even if a delete
            // path is added later without remembering to clean up.
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [Index("conversationId")],
)
data class MessageEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val conversationId: Long,
    val role: String,
    val content: String,
    val createdAtMillis: Long = System.currentTimeMillis(),
    /** Decode rate for assistant turns; zero when not applicable. */
    val tokensPerSecond: Double = 0.0,
    val tokenCount: Int = 0,
)

@Dao
interface ChatDao {

    @Query("SELECT * FROM conversations ORDER BY updatedAtMillis DESC")
    fun observeConversations(): Flow<List<ConversationEntity>>

    @Query("SELECT * FROM conversations WHERE id = :id")
    suspend fun conversation(id: Long): ConversationEntity?

    @Query("SELECT * FROM messages WHERE conversationId = :id ORDER BY id ASC")
    fun observeMessages(id: Long): Flow<List<MessageEntity>>

    @Query("SELECT * FROM messages WHERE conversationId = :id ORDER BY id ASC")
    suspend fun messages(id: Long): List<MessageEntity>

    @Insert
    suspend fun insert(conversation: ConversationEntity): Long

    @Insert
    suspend fun insert(message: MessageEntity): Long

    @Update
    suspend fun update(conversation: ConversationEntity)

    @Query("UPDATE messages SET content = :content, tokensPerSecond = :rate, tokenCount = :tokens WHERE id = :id")
    suspend fun updateMessage(id: Long, content: String, rate: Double, tokens: Int)

    @Query("DELETE FROM conversations WHERE id = :id")
    suspend fun deleteConversation(id: Long)

    @Query("DELETE FROM messages WHERE id = :id")
    suspend fun deleteMessage(id: Long)

    @Query("UPDATE conversations SET updatedAtMillis = :now WHERE id = :id")
    suspend fun touch(id: Long, now: Long = System.currentTimeMillis())

    /**
     * Removes conversations that never received a message.
     *
     * An earlier version created a row the moment a model finished loading, so a
     * user who opened the app several times without typing accumulated a list of
     * identical empty entries. New conversations are now only written once they
     * have content; this clears what the old behaviour left behind.
     */
    @Query(
        "DELETE FROM conversations WHERE id NOT IN " +
            "(SELECT DISTINCT conversationId FROM messages)",
    )
    suspend fun deleteEmptyConversations()
}

@Database(
    entities = [ConversationEntity::class, MessageEntity::class],
    version = 1,
    exportSchema = false,
)
abstract class ChatDatabase : RoomDatabase() {
    abstract fun chatDao(): ChatDao
}
