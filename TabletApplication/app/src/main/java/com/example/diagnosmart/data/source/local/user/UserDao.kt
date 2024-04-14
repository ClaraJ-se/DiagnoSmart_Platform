package com.example.diagnosmart.data.source.local.user

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Upsert
import kotlinx.coroutines.flow.Flow


/**
 * for doctors
 * */

@Dao
interface UserDao {
    @Query("SELECT * FROM user ORDER BY id")
    fun observeAll(): Flow<List<LocalUser>>

    @Upsert
    fun upsert(user: LocalUser)

    @Query("SELECT * FROM user WHERE position = :position ORDER BY id")
    fun getUserWithPosition(position: String): Flow<List<LocalUser>>

    @Query("SELECT * FROM user WHERE id = :userId")
    fun getUser(userId: Long): LocalUser

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(user: LocalUser): Long

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(users: List<LocalUser>)

    @Query("DELETE FROM user")
    suspend fun deleteAll()
}

