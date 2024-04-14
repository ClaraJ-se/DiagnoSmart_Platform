package com.example.diagnosmart.data.source.local.user

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey
import com.example.diagnosmart.data.source.models.User

@Entity(tableName = "user")
data class LocalUser(
        @PrimaryKey @ColumnInfo(name = "id") val id: Long = 0,
        val firstName: String? = null,
        val lastName: String? = null,
        val patronymic: String? = null,
        val additName: String? = null,
        val position: String? = null
)

fun LocalUser.toExternal() = User(
        firstName = firstName,
        lastName = lastName,
        patronymic = patronymic,
        additName = additName,
        position = position
)
