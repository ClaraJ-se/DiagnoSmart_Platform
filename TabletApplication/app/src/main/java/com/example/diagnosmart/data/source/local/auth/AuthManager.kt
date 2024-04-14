package com.example.diagnosmart.data.source.local.auth

import android.content.Context
import android.content.SharedPreferences

class AuthManager(context: Context) {
    companion object {
        private const val PREF_NAME = "inspector_prefs"
        private const val EMAIL = "email"
        private const val PASSWORD = "password"
    }

    private val sharedPreferences: SharedPreferences =
        context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)

    var email: String?
        get() = sharedPreferences.getString(EMAIL, null)
        set(value) {
            sharedPreferences.edit().putString(EMAIL, value).apply()
        }

    var password: String?
        get() = sharedPreferences.getString(PASSWORD, null)
        set(value) {
            sharedPreferences.edit().putString(PASSWORD, value).apply()
        }

    fun clearTokenData() {
        sharedPreferences.edit().remove(EMAIL).apply()
        sharedPreferences.edit().remove(PASSWORD).apply()
    }
}