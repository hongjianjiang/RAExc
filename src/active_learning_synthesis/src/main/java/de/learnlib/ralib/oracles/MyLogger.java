package de.learnlib.ralib.oracles;

public class MyLogger {
    static StringBuilder sb;
    static String logFileName;

    public static void init(String logFileName) {
        MyLogger.logFileName = logFileName;
        sb = new StringBuilder();
    }

    static void println(Object msg) {
        System.out.println(msg);
        sb.append(msg).append("\n");
    }

    static void println() {
        sb.append("\n");
    }

    public static String string() {
        return sb.toString();
    }
}