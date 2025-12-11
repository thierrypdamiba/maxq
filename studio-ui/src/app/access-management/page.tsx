"use client";

import { Users, Bell, Mail, MoreVertical, Plus, Copy } from "lucide-react";
import { useState } from "react";

export default function AccessManagementPage() {
  const [selectedRole, setSelectedRole] = useState("Base");

  return (
    <div className="p-10 max-w-7xl mx-auto">
      {/* Page Title */}
      <h1 className="text-3xl font-bold text-foreground mb-8">Access Management</h1>

      {/* Tabs */}
      <div className="border-b border-border-light mb-8">
        <div className="flex gap-8">
          <button className="pb-4 border-b-2 border-qdrant-purple text-foreground font-medium">
            User & Role Management
          </button>
          <button className="pb-4 text-foreground-muted hover:text-foreground">
            Cloud Management Keys
          </button>
        </div>
      </div>

      {/* User Management Card */}
      <div className="bg-surface rounded-lg p-6 mb-6">
        <div className="flex items-center gap-3 mb-2">
          <Users className="w-5 h-5 text-foreground" />
          <h2 className="text-lg font-semibold text-foreground">User Management</h2>
        </div>
        <p className="text-foreground-muted">
          Manage users, roles and invites for this account.
        </p>
      </div>

      {/* Send Invitation Section */}
      <div className="flex items-center gap-4 mb-6">
        <input
          type="email"
          placeholder="Send an invitation with the selected role by entering an email."
          className="flex-1 bg-surface border border-border-light rounded-md px-4 py-2 text-foreground placeholder-foreground-muted focus:outline-none focus:border-qdrant-purple"
        />
        <select
          value={selectedRole}
          onChange={(e) => setSelectedRole(e.target.value)}
          className="bg-surface border border-border-light rounded-md px-4 py-2 text-foreground focus:outline-none focus:border-qdrant-purple"
        >
          <option value="Base">Base</option>
          <option value="Admin">Admin</option>
          <option value="Owner">Owner</option>
        </select>
        <button className="btn-primary flex items-center gap-2">
          <Mail className="w-4 h-4" />
          Invite
        </button>
      </div>

      {/* User Table */}
      <div className="bg-surface rounded-lg border border-border-light overflow-hidden mb-8">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-surface-highlight border-b border-border-light">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-semibold text-foreground-muted uppercase tracking-wider">
                  EMAIL
                </th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-foreground-muted uppercase tracking-wider">
                  STATUS
                </th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-foreground-muted uppercase tracking-wider">
                  ROLES
                </th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-foreground-muted uppercase tracking-wider">
                  ACTIONS
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border-light">
              <tr className="hover:bg-surface-highlight/50 transition-colors">
                <td className="px-6 py-4">
                  <div className="flex items-center gap-2">
                    <span className="text-foreground">ptdamiba@gmail.com</span>
                    <span className="badge badge-purple text-xs">ME</span>
                    <span className="badge badge-success text-xs">OWNER</span>
                  </div>
                </td>
                <td className="px-6 py-4">
                  <span className="badge badge-success">ACTIVE</span>
                </td>
                <td className="px-6 py-4">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="badge badge-grey">QDRANTBASE</span>
                    <span className="badge badge-purple">ADMIN</span>
                    <span className="badge badge-grey w-6 h-6 rounded-full flex items-center justify-center text-xs">+1</span>
                  </div>
                </td>
                <td className="px-6 py-4">
                  <button className="text-foreground-muted hover:text-foreground">
                    <MoreVertical className="w-5 h-5" />
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        {/* Pagination */}
        <div className="px-6 py-4 border-t border-border-light flex items-center justify-between">
          <div className="flex items-center gap-2 text-sm text-foreground-muted">
            <span>Rows per page:</span>
            <select className="bg-surface border border-border-light rounded px-2 py-1 text-foreground">
              <option>10</option>
            </select>
          </div>
          <div className="text-sm text-foreground-muted">
            1-1 of 1
          </div>
        </div>
      </div>

      {/* Built-in Roles */}
      <div className="mb-8">
        <h3 className="text-lg font-semibold text-foreground mb-4">BUILT-IN ROLES</h3>
        <div className="space-y-3">
          <div className="bg-surface rounded-lg p-4">
            <div className="font-medium text-foreground mb-1">Owner</div>
            <div className="text-sm text-foreground-muted">Owner Role For An Account</div>
          </div>
          <div className="bg-surface rounded-lg p-4">
            <div className="font-medium text-foreground mb-1">Admin</div>
            <div className="text-sm text-foreground-muted">Description For Admin</div>
          </div>
          <div className="bg-surface rounded-lg p-4">
            <div className="font-medium text-foreground mb-1">Base</div>
            <div className="text-sm text-foreground-muted">Minimal Permissions Role...</div>
          </div>
        </div>
      </div>

      {/* Custom Roles */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-foreground">CUSTOM ROLES</h3>
          <button className="w-8 h-8 border border-border-light rounded flex items-center justify-center text-foreground-muted hover:text-foreground hover:border-foreground transition-colors">
            <Plus className="w-4 h-4" />
          </button>
        </div>
        <div className="bg-surface rounded-lg p-8 border border-border-light text-center">
          <div className="text-foreground-muted">NO ROLES FOUND</div>
        </div>
      </div>
    </div>
  );
}
