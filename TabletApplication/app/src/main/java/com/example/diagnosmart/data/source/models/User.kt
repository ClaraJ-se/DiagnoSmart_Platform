package com.example.diagnosmart.data.source.models

import com.example.diagnosmart.data.source.local.user.LocalUser

data class User(
        val firstName: String? = "",
        val lastName: String? = "",
        val patronymic: String? = "",
        val additName: String? = "",
        val position: String? = ""
)

fun User.toLocal() = LocalUser(
        firstName = firstName ?: "",
        lastName = lastName ?: "",
        patronymic = patronymic ?: "",
        additName = additName ?: "",
        position = position ?: ""
)

fun LocalUser.toExternal() = User(
        firstName = firstName,
        lastName = lastName,
        patronymic = patronymic,
        additName = additName,
        position = position
)